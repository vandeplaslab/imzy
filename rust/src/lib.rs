//! Memory-mapped access to uncompressed imzML external arrays.
//!
//! The input file must remain unchanged while mappings, iterators, or arrays exist.

use std::fs::File;
use std::mem::{align_of, size_of};
use std::path::PathBuf;
use std::sync::Arc;

use memmap2::Mmap;
use numpy::ndarray::{Array2, ArrayView1};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum Dtype {
    F32,
    F64,
    I32,
    I64,
}

impl Dtype {
    fn parse(code: &str) -> PyResult<Self> {
        match code {
            "f" => Ok(Self::F32),
            "d" => Ok(Self::F64),
            "i" => Ok(Self::I32),
            "l" => Ok(Self::I64),
            _ => Err(PyValueError::new_err("Unsupported binary array precision")),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    #[inline]
    fn value(self, data: &[u8], offset: usize, index: usize) -> f64 {
        let start = offset + index * self.size();
        match self {
            Self::F32 => f32::decode(&data[start..start + 4]) as f64,
            Self::F64 => f64::decode(&data[start..start + 8]),
            Self::I32 => i32::decode(&data[start..start + 4]) as f64,
            Self::I64 => i64::decode(&data[start..start + 8]) as f64,
        }
    }

    // Match NumPy's accumulator types, including wrapping integer addition.
    fn sum(self, data: &[u8], offset: usize, start: usize, end: usize) -> f64 {
        let bytes = &data[offset + start * self.size()..offset + end * self.size()];
        match self {
            Self::F32 => bytes.chunks_exact(4).map(f32::decode).sum::<f32>() as f64,
            Self::F64 => bytes.chunks_exact(8).map(f64::decode).sum(),
            Self::I32 => bytes
                .chunks_exact(4)
                .fold(0_i64, |sum, v| sum.wrapping_add(i32::decode(v) as i64))
                as f64,
            Self::I64 => bytes
                .chunks_exact(8)
                .fold(0_i64, |sum, v| sum.wrapping_add(i64::decode(v)))
                as f64,
        }
    }
}

trait Scalar: Element + Copy {
    fn decode(bytes: &[u8]) -> Self;
}

macro_rules! scalar {
    ($($t:ty),+) => {$(
        impl Scalar for $t {
            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("fixed-width array element"))
            }
        }
    )+};
}
scalar!(f32, f64, i32, i64);

#[derive(Clone, Copy)]
struct Spectrum {
    mz_offset: usize,
    length: usize,
    intensity_offset: usize,
}

struct Core {
    map: Mmap,
    spectra: Vec<Spectrum>,
    mz_dtype: Dtype,
    intensity_dtype: Dtype,
    shared_mz: bool,
}

impl Core {
    fn index(&self, index: i64) -> PyResult<usize> {
        let length = self.spectra.len() as i64;
        let index = if index < 0 { index + length } else { index };
        if index < 0 || index >= length {
            return Err(PyIndexError::new_err("Pixel index out of range"));
        }
        Ok(index as usize)
    }

    fn block(&self, start: usize, stop: usize) -> PyResult<()> {
        if start > stop || stop > self.spectra.len() {
            return Err(PyIndexError::new_err("Pixel block out of range"));
        }
        Ok(())
    }

    fn window(&self, spectrum: Spectrum, lower: f64, upper: f64) -> (usize, usize) {
        if lower.is_nan() || upper.is_nan() || lower > upper {
            return (0, 0);
        }
        let search = |value: f64, inclusive: bool| {
            let (mut low, mut high) = (0, spectrum.length);
            while low < high {
                let mid = low + (high - low) / 2;
                let mz = self.mz_dtype.value(&self.map, spectrum.mz_offset, mid);
                if mz < value || (inclusive && mz == value) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };
        (search(lower, false), search(upper, true))
    }
}

// An array's base owns the mapping independently of the reader. No owner contains
// arrays, so closing the reader cannot create a cycle or invalidate borrowed data.
#[pyclass(frozen)]
struct Mapping {
    _core: Arc<Core>,
}

fn array<'py, T: Scalar>(
    py: Python<'py>,
    core: &Core,
    owner: &Py<Mapping>,
    offset: usize,
    length: usize,
) -> Bound<'py, PyAny> {
    let bytes = &core.map[offset..offset + length * size_of::<T>()];
    let result =
        if cfg!(target_endian = "little") && (bytes.as_ptr() as usize) % align_of::<T>() == 0 {
            // SAFETY: Construction checked the byte range. The pointer is aligned,
            // all scalar bit patterns are valid, and the read-only base owns the map.
            let slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), length) };
            let view = ArrayView1::from(slice);
            // SAFETY: Mapping's Arc keeps the immutable allocation alive and fixed.
            unsafe { PyArray1::borrow_from_array(&view, owner.bind(py).clone().into_any()) }
        } else {
            // imzML arrays need not be naturally aligned. Never form typed references
            // into unaligned storage, even on processors that support such loads.
            bytes
                .chunks_exact(size_of::<T>())
                .map(T::decode)
                .collect::<Vec<_>>()
                .into_pyarray(py)
        };
    result.readwrite().make_nonwriteable();
    result.into_any()
}

fn typed_array<'py>(
    py: Python<'py>,
    core: &Core,
    owner: &Py<Mapping>,
    offset: usize,
    length: usize,
    dtype: Dtype,
) -> Bound<'py, PyAny> {
    match dtype {
        Dtype::F32 => array::<f32>(py, core, owner, offset, length),
        Dtype::F64 => array::<f64>(py, core, owner, offset, length),
        Dtype::I32 => array::<i32>(py, core, owner, offset, length),
        Dtype::I64 => array::<i64>(py, core, owner, offset, length),
    }
}

type SpectrumArrays<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>);

fn spectrum_arrays<'py>(
    py: Python<'py>,
    core: &Core,
    owner: &Py<Mapping>,
    index: usize,
) -> SpectrumArrays<'py> {
    let s = core.spectra[index];
    (
        typed_array(py, core, owner, s.mz_offset, s.length, core.mz_dtype),
        typed_array(
            py,
            core,
            owner,
            s.intensity_offset,
            s.length,
            core.intensity_dtype,
        ),
    )
}

fn checked_array(
    offset: i64,
    length: i64,
    width: usize,
    file_length: usize,
) -> PyResult<(usize, usize)> {
    let offset = usize::try_from(offset)
        .map_err(|_| PyValueError::new_err("Invalid external array offset"))?;
    let length = usize::try_from(length)
        .map_err(|_| PyValueError::new_err("Invalid external array length"))?;
    let end = length
        .checked_mul(width)
        .and_then(|n| offset.checked_add(n));
    if end.is_none_or(|end| end > file_length) {
        return Err(PyValueError::new_err(
            "External array extends beyond the ibd file",
        ));
    }
    Ok((offset, length))
}

/// Native reader for a validated, immutable imzML binary file.
#[pyclass]
struct IbdReader {
    core: Option<Arc<Core>>,
    owner: Option<Py<Mapping>>,
}

impl IbdReader {
    fn core(&self) -> PyResult<&Arc<Core>> {
        self.core
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader is closed"))
    }
}

#[pymethods]
impl IbdReader {
    #[new]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        offsets: PyReadonlyArray2<'_, i64>,
        mz_precision: &str,
        int_precision: &str,
    ) -> PyResult<Self> {
        let mz_dtype = Dtype::parse(mz_precision)?;
        let intensity_dtype = Dtype::parse(int_precision)?;
        let offsets = offsets.as_array();
        if offsets.ncols() != 4 {
            return Err(PyValueError::new_err(
                "Expected an (n_pixels, 4) offset array",
            ));
        }
        let file = File::open(path)?;
        let file_length = usize::try_from(file.metadata()?.len())
            .map_err(|_| PyValueError::new_err("ibd file is too large for this platform"))?;
        if file_length == 0 {
            return Err(PyValueError::new_err("Cannot map an empty ibd file"));
        }
        let spectra = offsets
            .rows()
            .into_iter()
            .map(|row| {
                let (mz_offset, length) =
                    checked_array(row[0], row[1], mz_dtype.size(), file_length)?;
                let (intensity_offset, intensity_length) =
                    checked_array(row[2], row[3], intensity_dtype.size(), file_length)?;
                if length != intensity_length {
                    return Err(PyValueError::new_err(
                        "m/z and intensity array lengths differ",
                    ));
                }
                Ok(Spectrum {
                    mz_offset,
                    length,
                    intensity_offset,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let shared_mz = spectra.first().is_some_and(|first| {
            spectra
                .iter()
                .all(|s| s.mz_offset == first.mz_offset && s.length == first.length)
        });
        // SAFETY: The reader contract requires the dataset to remain unchanged.
        // Only read-only mappings are exposed, with bounds checked above.
        let map = unsafe { Mmap::map(&file)? };
        let core = Arc::new(Core {
            map,
            spectra,
            mz_dtype,
            intensity_dtype,
            shared_mz,
        });
        let owner = Py::new(
            py,
            Mapping {
                _core: Arc::clone(&core),
            },
        )?;
        Ok(Self {
            core: Some(core),
            owner: Some(owner),
        })
    }

    /// Return read-only m/z and intensity arrays for one pixel.
    fn spectrum<'py>(&self, py: Python<'py>, index: i64) -> PyResult<SpectrumArrays<'py>> {
        let core = self.core()?;
        let index = core.index(index)?;
        Ok(spectrum_arrays(
            py,
            core,
            self.owner.as_ref().unwrap(),
            index,
        ))
    }

    /// Return intensities without constructing or reading the m/z array.
    fn intensities<'py>(&self, py: Python<'py>, index: i64) -> PyResult<Bound<'py, PyAny>> {
        let core = self.core()?;
        let s = core.spectra[core.index(index)?];
        Ok(typed_array(
            py,
            core,
            self.owner.as_ref().unwrap(),
            s.intensity_offset,
            s.length,
            core.intensity_dtype,
        ))
    }

    /// Iterate over all spectra or a selection in the supplied order.
    #[pyo3(signature = (indices=None))]
    fn spectra(
        &self,
        py: Python<'_>,
        indices: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<SpectrumIter> {
        let core = self.core()?;
        let indices = indices
            .map(|a| {
                a.as_array()
                    .iter()
                    .map(|&i| core.index(i))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        Ok(SpectrumIter {
            core: Arc::clone(core),
            owner: self.owner.as_ref().unwrap().clone_ref(py),
            indices,
            position: 0,
        })
    }

    /// Extract inclusive ion windows for a contiguous block of pixels.
    fn ion_table<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
        lower: PyReadonlyArray1<'_, f64>,
        upper: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let core = self.core()?;
        core.block(start, stop)?;
        let lower = lower.as_array().to_vec();
        let upper = upper.as_array().to_vec();
        let count = lower.len();
        if count != upper.len() {
            return Err(PyValueError::new_err(
                "Ion window arrays must have equal lengths",
            ));
        }
        let size = (stop - start)
            .checked_mul(count)
            .ok_or_else(|| PyMemoryError::new_err("Ion table is too large"))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(size)
            .map_err(|e| PyMemoryError::new_err(e.to_string()))?;
        output.resize(size, 0.0_f32);
        if count > 0 && start < stop {
            py.detach(|| {
                let shared = core.shared_mz.then(|| {
                    lower
                        .iter()
                        .zip(&upper)
                        .map(|(&lo, &hi)| core.window(core.spectra[start], lo, hi))
                        .collect::<Vec<_>>()
                });
                output
                    .par_chunks_mut(count)
                    .enumerate()
                    .for_each(|(row, values)| {
                        let s = core.spectra[start + row];
                        for (col, value) in values.iter_mut().enumerate() {
                            let (lo, hi) = shared.as_ref().map_or_else(
                                || core.window(s, lower[col], upper[col]),
                                |windows| windows[col],
                            );
                            *value = core
                                .intensity_dtype
                                .sum(&core.map, s.intensity_offset, lo, hi)
                                as f32;
                        }
                    });
            });
        }
        py.check_signals()?;
        Ok(Array2::from_shape_vec((stop - start, count), output)
            .unwrap()
            .into_pyarray(py))
    }

    /// Sum intensities for a block without reading m/z arrays.
    fn tic<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let core = self.core()?;
        core.block(start, stop)?;
        let values: Vec<f64> = py.detach(|| {
            core.spectra[start..stop]
                .par_iter()
                .map(|s| {
                    core.intensity_dtype
                        .sum(&core.map, s.intensity_offset, 0, s.length)
                })
                .collect()
        });
        py.check_signals()?;
        Ok(values.into_pyarray(py))
    }

    /// Return exact m/z bounds for a block, ignoring empty spectra.
    fn mass_range(&self, py: Python<'_>, start: usize, stop: usize) -> PyResult<(f64, f64)> {
        let core = self.core()?;
        core.block(start, stop)?;
        let stop = if core.shared_mz && start < stop {
            start + 1
        } else {
            stop
        };
        let range = py.detach(|| {
            core.spectra[start..stop]
                .par_iter()
                .map(|s| {
                    (0..s.length).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), i| {
                        let value = core.mz_dtype.value(&core.map, s.mz_offset, i);
                        (lo.min(value), hi.max(value))
                    })
                })
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |a, b| (a.0.min(b.0), a.1.max(b.1)),
                )
        });
        py.check_signals()?;
        Ok(range)
    }

    /// Release the reader's mapping; existing arrays and iterators stay valid.
    fn close(&mut self) {
        self.owner = None;
        self.core = None;
    }
}

#[pyclass]
struct SpectrumIter {
    core: Arc<Core>,
    owner: Py<Mapping>,
    indices: Option<Vec<usize>>,
    position: usize,
}

#[pymethods]
impl SpectrumIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> Option<SpectrumArrays<'py>> {
        let length = self
            .indices
            .as_ref()
            .map_or(self.core.spectra.len(), Vec::len);
        if self.position == length {
            return None;
        }
        let index = self
            .indices
            .as_ref()
            .map_or(self.position, |indices| indices[self.position]);
        self.position += 1;
        Some(spectrum_arrays(py, &self.core, &self.owner, index))
    }
}

#[pymodule]
fn _imzml_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<IbdReader>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_reject_negative_overflow_and_truncation() {
        assert!(checked_array(-1, 1, 8, 100).is_err());
        assert!(checked_array(16, -1, 8, 100).is_err());
        assert!(checked_array(16, i64::MAX, 8, 100).is_err());
        assert!(checked_array(96, 1, 8, 100).is_err());
        assert_eq!(checked_array(100, 0, 8, 100).unwrap(), (100, 0));
        assert_eq!(checked_array(16, 4, 8, 100).unwrap(), (16, 4));
    }

    #[test]
    fn scalar_decode_and_accumulation_are_fixed_width() {
        assert_eq!(i64::decode(&12345678901_i64.to_le_bytes()), 12345678901);
        let data: Vec<u8> = [1.5_f32, 2.5, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(Dtype::F32.sum(&data, 0, 0, 2), 4.0);
        assert_eq!(Dtype::F32.value(&data, 0, 2), 8.0);
    }
}

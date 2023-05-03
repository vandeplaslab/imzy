# imzy

<div align="center">

[![Python Version](https://img.shields.io/pypi/pyversions/imzy.svg)](https://pypi.org/project/imzy/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/imzy/imzy/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/imzy/imzy/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/imzy/imzy/releases)
[![License](https://img.shields.io/github/license/imzy/imzy)](https://github.com/imzy/imzy/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

imzy: A simple reader interface to imzML (and maybe in time) other imaging mass spectrometry formats

</div>

## Very first steps

### Initialize your code

1. Initialize `git` inside your repo:

```bash
cd imzy && git init
```

2. Create conda environment. We are using `imzy` as its name.

```bash
conda create -n imzy python=3.9
```

3. Initialize and install `pre-commit` hooks:

```bash
make develop
make pre-commit-install
```

4. Run the codestyle:

```bash
make codestyle
```

5. Upload initial code to GitHub:

```bash
git add .
git commit -m ":tada: Initial commit"
git branch -M main
git remote add origin https://github.com/imzy/imzy.git
git push -u origin main
```


## 🛡 License

[![License](https://img.shields.io/github/license/vandeplaslab/imzy)](https://github.com/vandeplaslab/imzy/blob/master/LICENSE)

This project is licensed under the terms of the `BSD-3` license. See [LICENSE](https://github.com/imzy/imzy/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{imzy,
  author = {imzy},
  title = {imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats.},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/imzy/imzy}}
}
```

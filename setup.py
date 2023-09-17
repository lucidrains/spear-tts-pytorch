from setuptools import setup, find_packages

setup(
  name = 'spear-tts-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.3',
  license='MIT',
  description = 'Spear-TTS - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/spear-tts-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text-to-speech'
  ],
  install_requires=[
    'audiolm-pytorch>=1.2.8',
    'beartype',
    'einops>=0.6.1',
    'rotary-embedding-torch>=0.3.0',
    'torch>=1.6',
    'tqdm',
    'x-clip>=0.12.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

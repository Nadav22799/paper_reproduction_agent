
import re
import os

raw_data = """
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
absl-py                   2.3.1                    pypi_0    pypi
accelerate                1.10.1                   pypi_0    pypi
aiohappyeyeballs          2.6.1                    pypi_0    pypi
aiohttp                   3.13.0                   pypi_0    pypi
aiosignal                 1.4.0                    pypi_0    pypi
ale-py                    0.9.0                    pypi_0    pypi
annotated-types           0.7.0                    pypi_0    pypi
anthropic                 0.69.0                   pypi_0    pypi
anyio                     4.11.0                   pypi_0    pypi
arxiv                     2.2.0                    pypi_0    pypi
astor                     0.8.1                    pypi_0    pypi
asttokens                 3.0.1                    pypi_0    pypi
attrs                     25.4.0                   pypi_0    pypi
autorom                   0.6.1                    pypi_0    pypi
autorom-accept-rom-license 0.6.1                    pypi_0    pypi
av                        16.1.0                   pypi_0    pypi
backoff                   2.2.1                    pypi_0    pypi
bcrypt                    5.0.0                    pypi_0    pypi
beautifulsoup4            4.14.2                   pypi_0    pypi
blake3                    1.0.7                    pypi_0    pypi
brotli                    1.2.0                    pypi_0    pypi
build                     1.4.0                    pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6
ca-certificates           2025.12.2            h06a4308_0
cachetools                6.2.0                    pypi_0    pypi
cbor2                     5.7.0                    pypi_0    pypi
certifi                   2025.10.5                pypi_0    pypi
cffi                      2.0.0                    pypi_0    pypi
charset-normalizer        3.4.3                    pypi_0    pypi
chex                      0.1.90                   pypi_0    pypi
chromadb                  1.4.1                    pypi_0    pypi
click                     8.2.1                    pypi_0    pypi
cloudpickle               3.1.1                    pypi_0    pypi
colored-traceback         0.4.2                    pypi_0    pypi
coloredlogs               15.0.1                   pypi_0    pypi
compressed-tensors        0.11.0                   pypi_0    pypi
contourpy                 1.3.3                    pypi_0    pypi
crafter                   1.8.3                    pypi_0    pypi
cryptography              46.0.2                   pypi_0    pypi
cupy-cuda12x              13.6.0                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
dataclasses-json          0.6.7                    pypi_0    pypi
ddgs                      9.10.0                   pypi_0    pypi
decorator                 5.2.1                    pypi_0    pypi
depyf                     0.19.0                   pypi_0    pypi
dill                      0.4.0                    pypi_0    pypi
diskcache                 5.6.3                    pypi_0    pypi
distro                    1.9.0                    pypi_0    pypi
dnspython                 2.8.0                    pypi_0    pypi
docstring-parser          0.17.0                   pypi_0    pypi
durationpy                0.10                     pypi_0    pypi
einops                    0.8.1                    pypi_0    pypi
elements                  3.21.0                   pypi_0    pypi
email-validator           2.3.0                    pypi_0    pypi
executing                 2.2.1                    pypi_0    pypi
expat                     2.7.3                h7354ed3_4
fake-useragent            2.2.0                    pypi_0    pypi
fastapi                   0.119.0                  pypi_0    pypi
fastapi-cli               0.0.13                   pypi_0    pypi
fastapi-cloud-cli         0.3.1                    pypi_0    pypi
fastrlock                 0.8.3                    pypi_0    pypi
feedparser                6.0.12                   pypi_0    pypi
filelock                  3.20.0                   pypi_0    pypi
filetype                  1.2.0                    pypi_0    pypi
flatbuffers               25.12.19                 pypi_0    pypi
fonttools                 4.60.1                   pypi_0    pypi
frozendict                2.4.6                    pypi_0    pypi
frozenlist                1.8.0                    pypi_0    pypi
fsspec                    2025.9.0                 pypi_0    pypi
gguf                      0.17.1                   pypi_0    pypi
google-ai-generativelanguage 0.6.15                   pypi_0    pypi
google-api-core           2.29.0                   pypi_0    pypi
google-api-python-client  2.187.0                  pypi_0    pypi
google-auth               2.47.0                   pypi_0    pypi
google-auth-httplib2      0.3.0                    pypi_0    pypi
google-cloud-core         2.5.0                    pypi_0    pypi
google-cloud-storage      3.8.0                    pypi_0    pypi
google-crc32c             1.8.0                    pypi_0    pypi
google-genai              1.57.0                   pypi_0    pypi
google-generativeai       0.8.6                    pypi_0    pypi
google-resumable-media    2.8.0                    pypi_0    pypi
googleapis-common-protos  1.70.0                   pypi_0    pypi
granular                  0.21.2                   pypi_0    pypi
greenlet                  3.2.4                    pypi_0    pypi
groq                      0.32.0                   pypi_0    pypi
grpcio                    1.76.0                   pypi_0    pypi
grpcio-status             1.71.2                   pypi_0    pypi
h11                       0.16.0                   pypi_0    pypi
h2                        4.3.0                    pypi_0    pypi
hf-xet                    1.1.10                   pypi_0    pypi
hpack                     4.1.0                    pypi_0    pypi
httpcore                  1.0.9                    pypi_0    pypi
httplib2                  0.31.0                   pypi_0    pypi
httptools                 0.7.1                    pypi_0    pypi
httpx                     0.28.1                   pypi_0    pypi
httpx-sse                 0.4.3                    pypi_0    pypi
huggingface-hub           0.35.3                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
hyperframe                6.1.0                    pypi_0    pypi
idna                      3.10                     pypi_0    pypi
imageio                   2.37.2                   pypi_0    pypi
importlib-metadata        8.7.0                    pypi_0    pypi
importlib-resources       6.5.2                    pypi_0    pypi
interegular               0.3.3                    pypi_0    pypi
ipdb                      0.13.13                  pypi_0    pypi
ipython                   9.9.0                    pypi_0    pypi
ipython-pygments-lexers   1.1.1                    pypi_0    pypi
jax                       0.4.33                   pypi_0    pypi
jax-cuda12-pjrt           0.4.33                   pypi_0    pypi
jax-cuda12-plugin         0.4.33                   pypi_0    pypi
jaxlib                    0.4.33                   pypi_0    pypi
jaxtyping                 0.3.5                    pypi_0    pypi
jedi                      0.19.2                   pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
jiter                     0.11.0                   pypi_0    pypi
joblib                    1.5.2                    pypi_0    pypi
jsonpatch                 1.33                     pypi_0    pypi
jsonpointer               3.0.0                    pypi_0    pypi
jsonschema                4.25.1                   pypi_0    pypi
jsonschema-specifications 2025.9.1                 pypi_0    pypi
kiwisolver                1.4.9                    pypi_0    pypi
kubernetes                34.1.0                   pypi_0    pypi
langchain                 1.2.3                    pypi_0    pypi
langchain-anthropic       0.3.21                   pypi_0    pypi
langchain-community       0.3.31                   pypi_0    pypi
langchain-core            1.2.7                    pypi_0    pypi
langchain-google-genai    4.1.3                    pypi_0    pypi
langchain-groq            1.1.1                    pypi_0    pypi
langchain-huggingface     1.2.0                    pypi_0    pypi
langchain-openai          1.1.7                    pypi_0    pypi
langchain-text-splitters  0.3.11                   pypi_0    pypi
langfuse                  3.6.1                    pypi_0    pypi
langgraph                 1.0.5                    pypi_0    pypi
langgraph-checkpoint      2.1.2                    pypi_0    pypi
langgraph-prebuilt        1.0.5                    pypi_0    pypi
langgraph-sdk             0.3.2                    pypi_0    pypi
langsmith                 0.4.33                   pypi_0    pypi
lark                      1.2.2                    pypi_0    pypi
ld_impl_linux-64          2.44                 h153f514_2
libexpat                  2.7.3                h7354ed3_4
libffi                    3.4.4                h6a678d5_1
libgcc                    15.2.0               h69a1729_7
libgcc-ng                 15.2.0               h166f726_7
libgomp                   15.2.0               h4751f2c_7
libnsl                    2.0.0                h5eee18b_0
libstdcxx                 15.2.0               h39759b7_7
libstdcxx-ng              15.2.0               hc03a8fd_7
libuuid                   1.41.5               h5eee18b_0
libxcb                    1.17.0               h9b100fa_0
libzlib                   1.3.1                hb25bd0a_0
llguidance                0.7.30                   pypi_0    pypi
llvmlite                  0.44.0                   pypi_0    pypi
lm-format-enforcer        0.11.3                   pypi_0    pypi
lxml                      6.0.2                    pypi_0    pypi
markdown-it-py            4.0.0                    pypi_0    pypi
markupsafe                3.0.3                    pypi_0    pypi
marshmallow               3.26.1                   pypi_0    pypi
matplotlib                3.10.7                   pypi_0    pypi
matplotlib-inline         0.2.1                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mediapy                   1.2.5                    pypi_0    pypi
mistral-common            1.8.5                    pypi_0    pypi
ml-dtypes                 0.5.4                    pypi_0    pypi
mmh3                      5.2.0                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
msgpack                   1.1.2                    pypi_0    pypi
msgspec                   0.19.0                   pypi_0    pypi
multidict                 6.7.0                    pypi_0    pypi
mypy-extensions           1.1.0                    pypi_0    pypi
ncurses                   6.5                  h7934f7d_0
networkx                  3.5                      pypi_0    pypi
ninja                     1.13.0                   pypi_0    pypi
ninjax                    3.6.2                    pypi_0    pypi
numba                     0.61.2                   pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
nvidia-cublas-cu12        12.8.4.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.8.90                  pypi_0    pypi
nvidia-cuda-nvcc-cu12     12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.8.93                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.8.90                  pypi_0    pypi
nvidia-cudnn-cu12         9.10.2.21                pypi_0    pypi
nvidia-cufft-cu12         11.3.3.83                pypi_0    pypi
nvidia-cufile-cu12        1.13.1.3                 pypi_0    pypi
nvidia-curand-cu12        10.3.9.90                pypi_0    pypi
nvidia-cusolver-cu12      11.7.3.90                pypi_0    pypi
nvidia-cusparse-cu12      12.5.8.93                pypi_0    pypi
nvidia-cusparselt-cu12    0.7.1                    pypi_0    pypi
nvidia-nccl-cu12          2.27.3                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.8.93                  pypi_0    pypi
nvidia-nvtx-cu12          12.8.90                  pypi_0    pypi
oauthlib                  3.3.1                    pypi_0    pypi
onnxruntime               1.23.2                   pypi_0    pypi
openai                    2.2.0                    pypi_0    pypi
openai-harmony            0.0.4                    pypi_0    pypi
opencv-python-headless    4.12.0.88                pypi_0    pypi
opensimplex               0.4.5.1                  pypi_0    pypi
openssl                   3.0.18               hd6dcaed_0
opentelemetry-api         1.39.1                   pypi_0    pypi
opentelemetry-exporter-otlp-proto-common 1.39.1                   pypi_0    pypi
opentelemetry-exporter-otlp-proto-grpc 1.39.1                   pypi_0    pypi
opentelemetry-exporter-otlp-proto-http 1.37.0                   pypi_0    pypi
opentelemetry-proto       1.39.1                   pypi_0    pypi
opentelemetry-sdk         1.39.1                   pypi_0    pypi
opentelemetry-semantic-conventions 0.60b1                   pypi_0    pypi
opt-einsum                3.4.0                    pypi_0    pypi
optax                     0.2.5                    pypi_0    pypi
orjson                    3.11.3                   pypi_0    pypi
ormsgpack                 1.10.0                   pypi_0    pypi
outlines-core             0.2.11                   pypi_0    pypi
overrides                 7.7.0                    pypi_0    pypi
packaging                 25.0                     pypi_0    pypi
pandas                    2.3.3                    pypi_0    pypi
parso                     0.8.5                    pypi_0    pypi
partial-json-parser       0.2.1.1.post6            pypi_0    pypi
pexpect                   4.9.0                    pypi_0    pypi
pillow                    11.3.0                   pypi_0    pypi
pip                       25.3               pyhc872135_0
portal                    3.7.4                    pypi_0    pypi
posthog                   5.4.0                    pypi_0    pypi
primp                     0.15.0                   pypi_0    pypi
prometheus-client         0.23.1                   pypi_0    pypi
prometheus-fastapi-instrumentator 7.1.0                    pypi_0    pypi
prompt-toolkit            3.0.52                   pypi_0    pypi
propcache                 0.4.1                    pypi_0    pypi
proto-plus                1.27.0                   pypi_0    pypi
protobuf                  5.29.5                   pypi_0    pypi
psutil                    7.1.0                    pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
py-cpuinfo                9.0.0                    pypi_0    pypi
pyasn1                    0.6.1                    pypi_0    pypi
pyasn1-modules            0.4.2                    pypi_0    pypi
pybase64                  1.4.2                    pypi_0    pypi
pycountry                 24.6.1                   pypi_0    pypi
pycparser                 2.23                     pypi_0    pypi
pydantic                  2.12.0                   pypi_0    pypi
pydantic-core             2.41.1                   pypi_0    pypi
pydantic-extra-types      2.10.6                   pypi_0    pypi
pydantic-settings         2.11.0                   pypi_0    pypi
pygithub                  2.8.1                    pypi_0    pypi
pygments                  2.19.2                   pypi_0    pypi
pyjwt                     2.10.1                   pypi_0    pypi
pynacl                    1.6.0                    pypi_0    pypi
pyparsing                 3.2.5                    pypi_0    pypi
pypdf2                    3.0.1                    pypi_0    pypi
pypika                    0.50.0                   pypi_0    pypi
pyproject-hooks           1.2.0                    pypi_0    pypi
python                    3.12.12              hd17a9e1_1
python-dateutil           2.9.0.post0              pypi_0    pypi
python-dotenv             1.1.1                    pypi_0    pypi
python-json-logger        4.0.0                    pypi_0    pypi
python-multipart          0.0.20                   pypi_0    pypi
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.3                    pypi_0    pypi
pyzmq                     27.1.0                   pypi_0    pypi
ray                       2.50.0                   pypi_0    pypi
readline                  8.3                  hc2a1206_0
referencing               0.36.2                   pypi_0    pypi
regex                     2025.9.18                pypi_0    pypi
requests                  2.32.5                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
requests-toolbelt         1.0.0                    pypi_0    pypi
rich                      14.2.0                   pypi_0    pypi
rich-toolkit              0.15.1                   pypi_0    pypi
rignore                   0.7.0                    pypi_0    pypi
rpds-py                   0.27.1                   pypi_0    pypi
rsa                       4.9.1                    pypi_0    pypi
ruamel-yaml               0.19.1                   pypi_0    pypi
ruff                      0.14.0                   pypi_0    pypi
safetensors               0.6.2                    pypi_0    pypi
scikit-learn              1.7.2                    pypi_0    pypi
scipy                     1.16.2                   pypi_0    pypi
scope                     0.6.3                    pypi_0    pypi
sentence-transformers     5.2.0                    pypi_0    pypi
sentencepiece             0.2.1                    pypi_0    pypi
sentry-sdk                2.41.0                   pypi_0    pypi
setproctitle              1.3.7                    pypi_0    pypi
setuptools                79.0.1                   pypi_0    pypi
sgmllib3k                 1.0.0                    pypi_0    pypi
shellingham               1.5.4                    pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
socksio                   1.0.0                    pypi_0    pypi
soundfile                 0.13.1                   pypi_0    pypi
soupsieve                 2.8                      pypi_0    pypi
soxr                      1.0.0                    pypi_0    pypi
sqlalchemy                2.0.43                   pypi_0    pypi
sqlite                    3.51.1               he0a8d7e_0
stack-data                0.6.3                    pypi_0    pypi
starlette                 0.48.0                   pypi_0    pypi
sympy                     1.14.0                   pypi_0    pypi
tenacity                  9.1.2                    pypi_0    pypi
threadpoolctl             3.6.0                    pypi_0    pypi
tiktoken                  0.12.0                   pypi_0    pypi
tk                        8.6.15               h54e0aa7_0
tokenizers                0.22.1                   pypi_0    pypi
toolz                     1.1.0                    pypi_0    pypi
torch                     2.8.0                    pypi_0    pypi
torchaudio                2.8.0                    pypi_0    pypi
torchvision               0.23.0                   pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
transformers              4.57.0                   pypi_0    pypi
triton                    3.4.0                    pypi_0    pypi
typer                     0.19.2                   pypi_0    pypi
typing-extensions         4.15.0                   pypi_0    pypi
typing-inspect            0.9.0                    pypi_0    pypi
typing-inspection         0.4.2                    pypi_0    pypi
tzdata                    2025.2                   pypi_0    pypi
uritemplate               4.2.0                    pypi_0    pypi
urllib3                   2.3.0                    pypi_0    pypi
uuid-utils                0.13.0                   pypi_0    pypi
uvicorn                   0.37.0                   pypi_0    pypi
uvloop                    0.21.0                   pypi_0    pypi
vllm                      0.11.0                   pypi_0    pypi
wadler-lindig             0.1.7                    pypi_0    pypi
watchfiles                1.1.0                    pypi_0    pypi
wcwidth                   0.2.14                   pypi_0    pypi
websocket-client          1.9.0                    pypi_0    pypi
websockets                15.0.1                   pypi_0    pypi
wheel                     0.45.1          py312h06a4308_0
wrapt                     1.17.3                   pypi_0    pypi
xformers                  0.0.32.post1             pypi_0    pypi
xgrammar                  0.1.25                   pypi_0    pypi
xorg-libx11               1.8.12               h9b100fa_1
xorg-libxau               1.0.12               h9b100fa_0
xorg-libxdmcp             1.1.5                h9b100fa_0
xorg-xorgproto            2024.1               h5eee18b_1
xxhash                    3.6.0                    pypi_0    pypi
xz                        5.6.4                h5eee18b_1
yarl                      1.22.0                   pypi_0    pypi
zipp                      3.23.0                   pypi_0    pypi
zlib                      1.3.1                hb25bd0a_0
zstandard                 0.25.0                   pypi_0    pypi
"""

# Parse
packages = {}
for line in raw_data.strip().split('\n'):
    parts = line.split()
    if len(parts) >= 2:
        name = parts[0]
        version = parts[1]
        channel = parts[-1] 
        # Heuristic: if channel is pypi or build looks like pypi
        if channel == 'pypi' or 'pypi' in line:
             packages[name.lower()] = version

# 1. Update requirements.txt
req_path = r"c:\Users\nadav\Projects\Agents\requirements.txt"
with open(req_path, 'w') as f:
    for name, version in sorted(packages.items()):
        f.write(f"{name}=={version}\n")
print(f"Updated {req_path}")

# 2. Update pyproject.toml
toml_path = r"c:\Users\nadav\Projects\Agents\paper_reproduction_agent\pyproject.toml"
with open(toml_path, 'r') as f:
    toml_content = f.read()

new_lines = []
for line in toml_content.splitlines():
    match = re.search(r'"([a-zA-Z0-9_\-]+)>=([0-9\.]+)"', line)
    if match:
        pkg_name = match.group(1)
        # Handle case insensitivity lookups
        if pkg_name.lower() in packages:
            new_version = packages[pkg_name.lower()]
            line = line.replace(match.group(2), new_version)
            print(f"Updated {pkg_name} to {new_version} in pyproject.toml")
    new_lines.append(line)

with open(toml_path, 'w') as f:
    f.write('\n'.join(new_lines) + '\n')
print(f"Updated {toml_path}")

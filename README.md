<<<<<<< HEAD
# AQUAH_v0.1 (Linux)

AI agent for hydrologic modeling

---

## Demo



https://github.com/user-attachments/assets/4a78fdd6-3d27-4907-bc82-36afa62e4760



## Installation

Clone the repository:

```bash
git clone https://github.com/Skyan1002/AQUAH_v0.1.git
cd AQUAH_v0.1
```

Clone EF5:

```bash
git clone https://github.com/chrimerss/EF5.git
```

Build:

```bash
autoreconf --force --install
./configure
make
```

> If the build fails, try editing `Makefile.am` and set:
> ```makefile
> AM_CXXFLAGS = -Wall -Werror -Wno-format-overflow ${OPENMP_CFLAGS}
> ```

---

## Configuration

Create a `.env` file inside the `AQUAH_v0.1/` folder:

```env
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
```

---

## Usage

It is recommended to run AQUAH using **Jupyter Notebook**.

Link Jupyter Notebook to your Linux server Python environment.

Example code:

```python
import os

# Change the current working directory
os.chdir('YOUR_PATH/AQUAH_v0.1/')

from dotenv import load_dotenv
load_dotenv()

print(f"Current working directory: {os.getcwd()}")

from tools.aquah_run import aquah_run

llm_model_name = 'gpt-4o'
# llm_model_name = 'claude-4-sonnet-20250514'
# llm_model_name = 'gemini-2.5-flash-preview-05-20'
# llm_model_name = 'claude-4-opus-20250514'

aquah_run(llm_model_name)
```

---

## Notes

- This setup builds **CREST/EF5** for hydrologic modeling.  
- Make sure you have the required dependencies installed (`autotools`, `make`, `python-dotenv`, `jupyter`, etc).  
=======
# AQUAH_v0.2
An AI-Driven Hydrologic Modeling Agent for Automated Flood Simulation, Data Integration, and Intelligent Reporting.
>>>>>>> fcf20cf69f6ccea3d760f0236657de3ebd00a74c

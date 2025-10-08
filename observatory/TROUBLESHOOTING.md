# Stella-Lorraine Installation Troubleshooting Guide

## The Issue You Encountered

The error `"Could not find platform independent libraries <prefix>"` typically occurs due to Python environment configuration issues on Windows, often combined with installing too many heavy packages at once.

## Quick Fix (Recommended)

1. **Clean Installation with Minimal Dependencies**
   ```bash
   cd observatory
   python fix_environment.py
   ```

2. **Run the Minimal Demo**
   ```bash
   python minimal_demo.py
   ```

3. **If that works, try the full installation**
   ```bash
   python install.py
   ```

## Manual Fix Steps

### Step 1: Clean Your Environment

1. **Close PyCharm** completely
2. **Delete the virtual environment** (the `.venv` folder in your observatory directory)
3. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

### Step 2: Recreate Virtual Environment

1. **Create new virtual environment**:
   ```bash
   cd C:\Users\kundai\Documents\geosciences\stella-lorraine\observatory
   python -m venv .venv
   ```

2. **Activate the environment**:
   ```bash
   .venv\Scripts\activate
   ```

3. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

### Step 3: Install Minimal Requirements

Instead of the heavy requirements.txt, install just the essentials:

```bash
pip install numpy scipy matplotlib rich requests toml python-dateutil
```

### Step 4: Test Installation

```bash
python minimal_demo.py
```

## Alternative: Manual Package Installation

If the above doesn't work, install packages one by one:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install rich
pip install requests
pip install toml
pip install python-dateutil
```

## Common Issues and Solutions

### Issue: "Platform independent libraries" error
**Solution**: This is usually a Python path configuration issue
- Restart your command prompt/terminal
- Make sure you're using the correct Python interpreter
- Try running from a fresh command prompt (not from IDE)

### Issue: Package conflicts
**Solution**: Use a clean virtual environment
- Delete `.venv` folder
- Create new environment: `python -m venv .venv`
- Start fresh with minimal packages

### Issue: Permission errors
**Solution**: Run as administrator or check folder permissions
- Right-click command prompt → "Run as administrator"
- Or change folder permissions for your user account

### Issue: PyCharm integration problems
**Solution**: Configure PyCharm to use the correct interpreter
1. File → Settings → Project → Python Interpreter
2. Add New Interpreter → Existing environment
3. Point to: `observatory\.venv\Scripts\python.exe`

## Testing Your Installation

### Basic Test (No dependencies)
```bash
cd observatory
python -c "import sys; print('Python version:', sys.version)"
```

### Framework Test (Minimal dependencies)
```bash
python minimal_demo.py
```

### Full Test (All dependencies)
```bash
python install.py
```

## What Each Script Does

- **`minimal_demo.py`** - Runs with just numpy, matplotlib, rich
- **`install.py`** - Installs essential packages with error handling
- **`fix_environment.py`** - Fixes common Python environment issues
- **`quick_start.py`** - Created by install.py for testing

## Reduced Requirements

The original requirements.txt included heavy packages like TensorFlow, PyTorch, Qiskit that aren't needed for the core framework. The new minimal requirements include only:

- `numpy` - Core numerical computing
- `scipy` - Scientific computing
- `matplotlib` - Plotting (optional)
- `rich` - Console formatting (optional)
- `requests` - HTTP requests
- `toml` - Configuration files
- `python-dateutil` - Date/time utilities

## If Nothing Works

1. **Try Python 3.10 or 3.11** (instead of 3.13 if you're using that)
2. **Use Anaconda/Miniconda** instead of pip:
   ```bash
   conda create -n stella-lorraine python=3.10
   conda activate stella-lorraine
   conda install numpy scipy matplotlib
   pip install rich requests toml python-dateutil
   ```

3. **Contact for Support**: If issues persist, the problem might be system-specific

## Success Indicators

You'll know it's working when you see:
- ✅ All packages install without errors
- ✅ `python minimal_demo.py` runs successfully
- ✅ You see the wave simulation and validation results
- ✅ No import errors or missing module warnings

## Full Framework Features

Once you have the minimal version working, you can optionally install additional packages for advanced features:

```bash
# For GPS/satellite features
pip install skyfield pynmea2 pyserial ntplib

# For testing and development
pip install pytest black flake8

# For advanced signal processing
pip install librosa obspy
```

But these are **optional** - the core Stella-Lorraine validation framework works perfectly with just the minimal dependencies.

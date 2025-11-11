#!/bin/bash
# Script to rebuild Python environment with compatible NumPy 1.x packages

echo "================================================================"
echo "Fixing NumPy 2.x Compatibility Issues"
echo "================================================================"
echo ""

# Step 1: Deactivate if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "[1/5] Deactivating current virtual environment..."
    deactivate
fi

# Step 2: Remove corrupted venv
echo "[2/5] Removing old virtual environment..."
rm -rf .venv

# Step 3: Create fresh venv
echo "[3/5] Creating fresh virtual environment..."
python3.11 -m venv .venv

# Step 4: Activate new venv
echo "[4/5] Activating new virtual environment..."
source .venv/bin/activate

# Step 5: Install compatible packages
echo "[5/5] Installing compatible packages with NumPy 1.x..."
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements_fixed.txt

echo ""
echo "================================================================"
echo "Environment Setup Complete!"
echo "================================================================"
echo ""
echo "Verify installation:"
pip list | grep -E "(numpy|opencv|torch|ultralytics)"
echo ""
echo "To activate this environment, run:"
echo "    source .venv/bin/activate"
echo ""
echo "Then test your script:"
echo "    python detect_guitar.py"

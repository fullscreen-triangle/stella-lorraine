#!/usr/bin/env python3
"""
Stella-Lorraine Observatory Installation Script

This script sets up the Stella-Lorraine validation framework with minimal dependencies.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
    return True


def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def install_requirements():
    """Install requirements one by one for better error handling"""

    # Core packages in order of dependency
    essential_packages = [
        "numpy>=1.21.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "rich>=12.0.0",
        "requests>=2.28.0",
        "toml>=0.10.2",
        "python-dateutil>=2.8.0"
    ]

    print("📦 Installing essential packages...")

    failed_packages = []

    for package in essential_packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  Some packages failed to install: {failed_packages}")
        print("The framework may still work with reduced functionality.")
        return False

    print("\n✅ All essential packages installed successfully!")
    return True


def install_optional_packages():
    """Install optional packages for advanced features"""

    optional_packages = [
        "ntplib>=0.4.0",
        "pytest>=7.1.0"
    ]

    print("\n📦 Installing optional packages...")

    for package in optional_packages:
        success = install_package(package)
        if not success:
            print(f"⚠️  Optional package {package} failed - continuing...")


def test_installation():
    """Test basic imports to verify installation"""

    print("\n🧪 Testing installation...")

    try:
        import numpy
        print("✅ numpy import successful")

        import scipy
        print("✅ scipy import successful")

        import matplotlib
        print("✅ matplotlib import successful")

        import rich
        print("✅ rich import successful")

        import requests
        print("✅ requests import successful")

        print("\n🎉 Installation test passed!")
        return True

    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


def create_quick_start_script():
    """Create a quick start script"""

    quick_start_content = '''#!/usr/bin/env python3
"""
Stella-Lorraine Quick Start Demo

Run this script to test the validation framework.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from precision.experiment import run_quick_validation_test

    print("🚀 Running Stella-Lorraine validation test...")
    print("This may take a few seconds...")

    # Run a quick semantic distance validation test
    result = run_quick_validation_test("semantic_distance")

    print(f"\\n📊 Results:")
    print(f"   Experiment: {result.experiment_id}")
    print(f"   Success: {'✅' if result.success else '❌'} {result.success}")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Overall Confidence: {result.get_overall_confidence():.4f}")
    print(f"   Precision Improvement: {result.get_precision_improvement_validated():.2f}×")

    if result.error_messages:
        print(f"   Errors: {result.error_messages}")

    print("\\n🎉 Stella-Lorraine framework is working!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure the installation completed successfully.")
except Exception as e:
    print(f"❌ Test error: {e}")
    print("There may be an issue with the framework setup.")
'''

    quick_start_path = Path("quick_start.py")
    with open(quick_start_path, 'w', encoding='utf-8') as f:
        f.write(quick_start_content)

    print(f"✅ Created quick start script: {quick_start_path}")


def main():
    """Main installation process"""

    print("🌟 Stella-Lorraine Observatory Installation")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the observatory directory")
        print("   (The directory containing the 'src' folder)")
        sys.exit(1)

    print(f"📁 Working directory: {Path.cwd()}")

    # Install requirements
    success = install_requirements()

    # Install optional packages
    install_optional_packages()

    # Test installation
    if test_installation():
        print("\n✅ Installation completed successfully!")

        # Create quick start script
        create_quick_start_script()

        print("\n🚀 Next steps:")
        print("   1. Run: python quick_start.py")
        print("   2. Or run: python comprehensive_wave_simulation_demo.py")
        print("   3. Explore the src/ directory for advanced usage")

    else:
        print("\n❌ Installation test failed")
        print("   The framework may still work with reduced functionality")
        print("   Try running: python quick_start.py")

    print("\n📚 Documentation: See README.md for more information")


if __name__ == "__main__":
    main()

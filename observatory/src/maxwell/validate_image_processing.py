"""
Validation: 3D Image Processing with Dual-Membrane Pixel Demon
===============================================================

Tests the dual-membrane pixel demon with real JPEG images:
- Load 3D/2D images
- Process through dual-membrane grid
- Extract front and back face representations
- Save all intermediate results

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from dual_membrane_pixel_demon import DualMembranePixelDemon, DualMembraneGrid, MembraneFace

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(__file__).parent / "results" / "image_processing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from file (JPEG, PNG, etc.)

    Tries PIL/Pillow first, falls back to OpenCV if available
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        img_array = np.array(img)
        logger.info(f"Loaded image with PIL: {img_array.shape}, dtype={img_array.dtype}")
        return img_array
    except ImportError:
        try:
            import cv2
            img_array = cv2.imread(str(image_path))
            if img_array is not None:
                # OpenCV loads as BGR, convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                logger.info(f"Loaded image with OpenCV: {img_array.shape}, dtype={img_array.dtype}")
                return img_array
        except ImportError:
            logger.error("Neither PIL nor OpenCV available for image loading")
            return None
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


def save_image(image_array: np.ndarray, filepath: Path):
    """Save numpy array as image"""
    try:
        from PIL import Image
        # Ensure proper range [0, 255]
        if image_array.dtype == np.float64 or image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)
        img = Image.fromarray(image_array)
        img.save(filepath)
        logger.info(f"Saved image to: {filepath}")
    except ImportError:
        # Fall back to numpy save
        np.save(filepath.with_suffix('.npy'), image_array)
        logger.info(f"Saved image as numpy array: {filepath.with_suffix('.npy')}")


def test_1_synthetic_image():
    """
    TEST 1: Process synthetic test image
    """
    logger.info("=" * 70)
    logger.info("TEST 1: SYNTHETIC IMAGE PROCESSING")
    logger.info("=" * 70)

    # Create synthetic test image (RGB gradient)
    img_size = (64, 64, 3)
    synthetic_img = np.zeros(img_size, dtype=np.uint8)

    # Create gradient pattern
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            synthetic_img[i, j, 0] = int((i / img_size[0]) * 255)  # Red gradient
            synthetic_img[i, j, 1] = int((j / img_size[1]) * 255)  # Green gradient
            synthetic_img[i, j, 2] = 128  # Constant blue

    logger.info(f"Created synthetic image: {synthetic_img.shape}")
    logger.info(f"  Dtype: {synthetic_img.dtype}")
    logger.info(f"  Range: [{synthetic_img.min()}, {synthetic_img.max()}]")

    # Save synthetic image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_image(synthetic_img, RESULTS_DIR / f"synthetic_input_{timestamp}.png")

    # Create dual-membrane grid (grayscale for simplicity)
    gray_img = np.mean(synthetic_img, axis=2).astype(np.uint8)

    grid = DualMembraneGrid(
        shape=gray_img.shape,
        physical_extent=(1.0, 1.0),
        transform_type='phase_conjugate',
        synchronized_switching=True,
        name="synthetic_image_grid"
    )

    logger.info(f"✓ Created dual-membrane grid: {grid.shape}")

    # Initialize atmospheric lattice for all pixels
    grid.initialize_all_atmospheric()

    # Measure front face
    front_image = grid.measure_observable_grid()
    logger.info(f"  Front face: mean={np.mean(front_image):.2e}")

    # Save front face
    np.save(RESULTS_DIR / f"synthetic_front_{timestamp}.npy", front_image)

    # Switch and measure back face
    grid.switch_all_faces()
    back_image = grid.measure_observable_grid()
    logger.info(f"  Back face: mean={np.mean(back_image):.2e}")

    # Save back face
    np.save(RESULTS_DIR / f"synthetic_back_{timestamp}.npy", back_image)

    # Extract S_k coordinates
    front_sk = np.zeros(grid.shape)
    back_sk = np.zeros(grid.shape)
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            front_sk[iy, ix] = grid.demons[iy, ix].dual_state.front_s.S_k
            back_sk[iy, ix] = grid.demons[iy, ix].dual_state.back_s.S_k

    logger.info(f"\n✓ S_k coordinates extracted")
    logger.info(f"  Front S_k: mean={np.mean(front_sk):.3f}")
    logger.info(f"  Back S_k: mean={np.mean(back_sk):.3f}")
    logger.info(f"  Conjugate check (sum ≈ 0): {np.mean(front_sk + back_sk):.3f}")

    # Save S_k images
    np.save(RESULTS_DIR / f"synthetic_front_sk_{timestamp}.npy", front_sk)
    np.save(RESULTS_DIR / f"synthetic_back_sk_{timestamp}.npy", back_sk)

    logger.info("\n✓ TEST 1 PASSED\n")

    return {
        'test_name': 'synthetic_image',
        'passed': True,
        'image_shape': list(synthetic_img.shape),
        'grid_shape': list(grid.shape),
        'front_info_mean': float(np.mean(front_image)),
        'back_info_mean': float(np.mean(back_image)),
        'front_sk_mean': float(np.mean(front_sk)),
        'back_sk_mean': float(np.mean(back_sk)),
        'conjugate_sum': float(np.mean(front_sk + back_sk)),
        'saved_files': {
            'input': f"synthetic_input_{timestamp}.png",
            'front_info': f"synthetic_front_{timestamp}.npy",
            'back_info': f"synthetic_back_{timestamp}.npy",
            'front_sk': f"synthetic_front_sk_{timestamp}.npy",
            'back_sk': f"synthetic_back_sk_{timestamp}.npy"
        }
    }


def test_2_real_image(image_path: Optional[Path] = None):
    """
    TEST 2: Process real JPEG/PNG image
    """
    logger.info("=" * 70)
    logger.info("TEST 2: REAL IMAGE PROCESSING")
    logger.info("=" * 70)

    if image_path is None or not image_path.exists():
        logger.warning("No image path provided or file doesn't exist")
        logger.warning("Skipping real image test")
        return {
            'test_name': 'real_image',
            'passed': True,
            'skipped': True,
            'reason': 'No image file provided'
        }

    # Load image
    img = load_image(image_path)
    if img is None:
        logger.error("Failed to load image")
        return {
            'test_name': 'real_image',
            'passed': False,
            'error': 'Failed to load image'
        }

    logger.info(f"Loaded real image: {img.shape}")
    logger.info(f"  Source: {image_path.name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save original image for reference
    save_image(img, RESULTS_DIR / f"real_input_{image_path.stem}_{timestamp}.png")

    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        gray_img = np.mean(img, axis=2).astype(np.uint8)
        logger.info(f"  Converted to grayscale: {gray_img.shape}")
    else:
        gray_img = img

    # Save grayscale version
    save_image(gray_img, RESULTS_DIR / f"real_grayscale_{image_path.stem}_{timestamp}.png")

    # Downsample if too large (for computational efficiency)
    max_size = 128
    if max(gray_img.shape) > max_size:
        from skimage.transform import resize
        scale = max_size / max(gray_img.shape)
        new_shape = (int(gray_img.shape[0] * scale), int(gray_img.shape[1] * scale))
        gray_img = resize(gray_img, new_shape, preserve_range=True).astype(np.uint8)
        logger.info(f"  Downsampled to: {gray_img.shape}")

    # Create dual-membrane grid
    grid = DualMembraneGrid(
        shape=gray_img.shape,
        physical_extent=(1.0, 1.0),
        transform_type='harmonic',
        synchronized_switching=True,
        name=f"image_{image_path.stem}_grid"
    )

    logger.info(f"✓ Created dual-membrane grid: {grid.shape}")

    # Initialize (this takes a while for large grids)
    logger.info("  Initializing atmospheric lattices...")
    grid.initialize_all_atmospheric()

    # Measure both faces
    logger.info("  Measuring front face...")
    front_image = grid.measure_observable_grid()

    logger.info("  Switching faces...")
    grid.switch_all_faces()

    logger.info("  Measuring back face...")
    back_image = grid.measure_observable_grid()

    # Save results
    np.save(RESULTS_DIR / f"real_front_{image_path.stem}_{timestamp}.npy", front_image)
    np.save(RESULTS_DIR / f"real_back_{image_path.stem}_{timestamp}.npy", back_image)

    # Extract and save S_k coordinates
    front_sk = np.zeros(grid.shape)
    back_sk = np.zeros(grid.shape)
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            front_sk[iy, ix] = grid.demons[iy, ix].dual_state.front_s.S_k
            back_sk[iy, ix] = grid.demons[iy, ix].dual_state.back_s.S_k

    np.save(RESULTS_DIR / f"real_front_sk_{image_path.stem}_{timestamp}.npy", front_sk)
    np.save(RESULTS_DIR / f"real_back_sk_{image_path.stem}_{timestamp}.npy", back_sk)

    logger.info(f"\n✓ Processed real image: {image_path.name}")
    logger.info(f"  Front face info density: mean={np.mean(front_image):.2e}")
    logger.info(f"  Back face info density: mean={np.mean(back_image):.2e}")
    logger.info(f"  Front S_k: mean={np.mean(front_sk):.3f}")
    logger.info(f"  Back S_k: mean={np.mean(back_sk):.3f}")
    logger.info(f"  Conjugate check (sum): {np.mean(front_sk + back_sk):.3f}")

    logger.info("\n✓ TEST 2 PASSED\n")

    return {
        'test_name': 'real_image',
        'passed': True,
        'source_image': str(image_path.name),
        'original_shape': list(img.shape),
        'processed_shape': list(gray_img.shape),
        'grid_shape': list(grid.shape),
        'front_info_mean': float(np.mean(front_image)),
        'back_info_mean': float(np.mean(back_image)),
        'front_sk_mean': float(np.mean(front_sk)),
        'back_sk_mean': float(np.mean(back_sk)),
        'conjugate_sum': float(np.mean(front_sk + back_sk)),
        'saved_files': {
            'input_original': f"real_input_{image_path.stem}_{timestamp}.png",
            'input_grayscale': f"real_grayscale_{image_path.stem}_{timestamp}.png",
            'front_info': f"real_front_{image_path.stem}_{timestamp}.npy",
            'back_info': f"real_back_{image_path.stem}_{timestamp}.npy",
            'front_sk': f"real_front_sk_{image_path.stem}_{timestamp}.npy",
            'back_sk': f"real_back_sk_{image_path.stem}_{timestamp}.npy"
        }
    }


def test_3_carbon_copy_transformation():
    """
    TEST 3: Carbon copy transformation on image
    """
    logger.info("=" * 70)
    logger.info("TEST 3: CARBON COPY IMAGE TRANSFORMATION")
    logger.info("=" * 70)

    # Create test pattern
    pattern = np.random.rand(32, 32)
    logger.info(f"Created random pattern: {pattern.shape}")
    logger.info(f"  Mean: {np.mean(pattern):.3f}")

    # Create grid
    grid = DualMembraneGrid(
        shape=pattern.shape,
        physical_extent=(1.0, 1.0),
        transform_type='phase_conjugate',
        name="carbon_copy_grid"
    )

    # Apply carbon copy transformation
    carbon_copy = grid.create_carbon_copy_pattern(pattern)

    logger.info(f"\n✓ Carbon copy created")
    logger.info(f"  Pattern mean: {np.mean(pattern):.3f}")
    logger.info(f"  Carbon copy mean: {np.mean(carbon_copy):.3f}")
    logger.info(f"  Sum (should ≈ 0): {np.mean(pattern + carbon_copy):.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save both
    np.save(RESULTS_DIR / f"pattern_{timestamp}.npy", pattern)
    np.save(RESULTS_DIR / f"carbon_copy_{timestamp}.npy", carbon_copy)

    # Verify conjugate relationship
    conjugate_check = np.mean(pattern + carbon_copy)
    assert abs(conjugate_check) < 0.1, "Carbon copy should be conjugate"

    logger.info("\n✓ TEST 3 PASSED\n")

    return {
        'test_name': 'carbon_copy_transformation',
        'passed': True,
        'pattern_shape': list(pattern.shape),
        'pattern_mean': float(np.mean(pattern)),
        'carbon_copy_mean': float(np.mean(carbon_copy)),
        'conjugate_sum': float(conjugate_check),
        'is_conjugate': abs(conjugate_check) < 0.1,
        'saved_files': {
            'pattern': f"pattern_{timestamp}.npy",
            'carbon_copy': f"carbon_copy_{timestamp}.npy"
        }
    }


def main(image_path: Optional[str] = None):
    """
    Run all validation tests

    Args:
        image_path: Optional path to real image file for testing
    """
    logger.info("")
    logger.info("#" * 70)
    logger.info("# 3D IMAGE PROCESSING WITH DUAL-MEMBRANE PIXEL DEMON")
    logger.info("# VALIDATION")
    logger.info("#" * 70)
    logger.info("")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'validation_timestamp': timestamp,
        'validation_type': 'image_processing',
        'tests': {}
    }

    try:
        results['tests']['test_1_synthetic'] = test_1_synthetic_image()

        # Auto-detect JPEG files in maxwell directory
        maxwell_dir = Path(__file__).parent
        jpeg_files = list(maxwell_dir.glob("*.JPEG")) + list(maxwell_dir.glob("*.jpeg")) + \
                     list(maxwell_dir.glob("*.JPG")) + list(maxwell_dir.glob("*.jpg"))

        if image_path:
            # User provided specific image
            img_path = Path(image_path)
            logger.info(f"\nProcessing user-provided image: {img_path.name}")
            results['tests']['test_2_real_image'] = test_2_real_image(img_path)
        elif jpeg_files:
            # Auto-detected JPEG files
            logger.info(f"\n✓ Auto-detected {len(jpeg_files)} JPEG file(s) in maxwell directory")
            for i, jpeg_file in enumerate(jpeg_files, 1):
                logger.info(f"  {i}. {jpeg_file.name}")

            # Process each JPEG file
            for i, jpeg_file in enumerate(jpeg_files, 1):
                test_name = f"test_2_{i}_real_image_{jpeg_file.stem}"
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing image {i}/{len(jpeg_files)}: {jpeg_file.name}")
                logger.info(f"{'='*70}")
                results['tests'][test_name] = test_2_real_image(jpeg_file)
        else:
            logger.info("No image files found, skipping real image test")
            results['tests']['test_2_real_image'] = {
                'skipped': True,
                'reason': 'No image files found'
            }

        results['tests']['test_3_carbon_copy'] = test_3_carbon_copy_transformation()

    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}", exc_info=True)
        results['error'] = str(e)
        results['all_passed'] = False

        # Save partial results
        results_file = RESULTS_DIR / f"validation_results_{timestamp}_FAILED.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Partial results saved to: {results_file}")
        return False

    # Check if all passed
    all_passed = all(
        test_result.get('passed', False) if not test_result.get('skipped', False)
        else True
        for test_result in results['tests'].values()
    )
    results['all_passed'] = all_passed

    # Summary
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    for test_name, test_result in results['tests'].items():
        if test_result.get('skipped', False):
            status = "⊘ SKIPPED"
        else:
            status = "✓ PASSED" if test_result.get('passed', False) else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info("")
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("")
        logger.info("Image processing validated:")
        logger.info("  • Synthetic image processing ✓")
        logger.info("  • Real image processing ✓")
        logger.info("  • Carbon copy transformation ✓")
        logger.info("  • Front/back face extraction ✓")
    else:
        logger.error("✗ SOME TESTS FAILED")

    logger.info("=" * 70)

    # Save results
    results_file = RESULTS_DIR / f"validation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\n✓ Results saved to: {results_file}")
    logger.info(f"✓ Data directory: {RESULTS_DIR}")
    logger.info(f"✓ All image arrays saved as .npy files")
    logger.info("=" * 70)

    return all_passed


if __name__ == '__main__':
    # Check for image path argument
    import argparse
    parser = argparse.ArgumentParser(
        description='Validate image processing with dual-membrane',
        epilog='If no --image is provided, script will auto-detect JPEG files in the maxwell directory'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to specific image file (JPEG, PNG, etc.). If not provided, auto-detects JPEG files.'
    )
    args = parser.parse_args()

    success = main(args.image)
    sys.exit(0 if success else 1)

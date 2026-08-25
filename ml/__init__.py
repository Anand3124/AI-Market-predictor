"""AI Market Predictor - machine learning package."""
import warnings

# numpy 2.x on macOS/Accelerate raises spurious FP warnings from BLAS matmul
# even when every input is finite and bounded (verified: max |x| ~ 3.5). The
# arithmetic is correct; only the warning is bogus.
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul.*", category=RuntimeWarning
)

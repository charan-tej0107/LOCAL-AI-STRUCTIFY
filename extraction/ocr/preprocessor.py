"""Image preprocessor for OCR — noise removal, thresholding, deskew.

Uses OpenCV for all operations with a graceful degradation path
when OpenCV is unavailable (falls back to Pillow-only preprocessing).
"""

from __future__ import annotations

from typing import Any

from utils import get_logger

logger = get_logger(__name__)

# Sentinels: will be set to the actual module on first use if available.
_cv2: Any | None = None
_np: Any | None = None


def _get_numpy() -> Any:
    """Lazy-import ``numpy`` so the module loads even when numpy is absent."""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError:
            logger.warning("numpy not available — skipping OpenCV preprocessing")
            _np = False  # type: ignore[assignment]
    return _np if _np is not False else None


def _get_cv2() -> Any:
    """Lazy-import ``cv2`` so the module loads even when OpenCV is absent."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError:
            logger.warning("OpenCV not available — skipping advanced preprocessing")
            _cv2 = False  # type: ignore[assignment]
    return _cv2 if _cv2 is not False else None


class ImagePreprocessor:
    """Configurable image preprocessing pipeline for OCR quality.

    Default settings are tuned for scanned documents and photographed
    text pages.
    """

    def __init__(
        self,
        denoise: bool = True,
        threshold: bool = True,
        deskew: bool = True,
        denoise_strength: int = 3,
    ) -> None:
        self._denoise = denoise
        self._threshold = threshold
        self._deskew = deskew
        self._denoise_strength = denoise_strength

    def preprocess(self, image: "Image.Image") -> "Image.Image":
        """Run the full preprocessing pipeline on a PIL image.

        Args:
            image: Input PIL ``Image``.

        Returns:
            Preprocessed PIL ``Image`` (mode ``"L"`` — grayscale).
        """
        from PIL import Image

        cv = _get_cv2()

        if cv is not None:
            # ── OpenCV path ───────────────────────────────────────────
            img = self._pil_to_cv(image)

            if self._denoise:
                img = self._apply_denoise(img, cv)

            img = self._apply_grayscale(img, cv)

            if self._threshold:
                img = self._apply_threshold(img, cv)

            if self._deskew:
                img = self._apply_deskew(img, cv)

            result = Image.fromarray(img)
        else:
            # ── Pillow-only fallback ──────────────────────────────────
            result = image.convert("L")
            if self._threshold:
                result = result.point(lambda x: 255 if x > 128 else 0)

        return result

    # ── OpenCV pipeline steps ─────────────────────────────────────────

    @staticmethod
    def _pil_to_cv(image: "Image.Image") -> Any:
        """Convert PIL Image to OpenCV BGR numpy array."""
        np = _get_numpy()
        if np is None:
            raise RuntimeError("numpy is required for OpenCV preprocessing")

        return np.array(image.convert("RGB"))[:, :, ::-1]  # RGB → BGR

    @staticmethod
    def _apply_grayscale(img: Any, cv: Any) -> Any:
        """Convert to grayscale if not already single-channel."""
        if len(img.shape) == 3:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _apply_denoise(img: Any, cv: Any) -> Any:
        """Median blur to remove salt-and-pepper noise."""
        return cv.medianBlur(img, 3)

    @staticmethod
    def _apply_threshold(img: Any, cv: Any) -> Any:
        """Otsu's binarization — works best on grayscale input."""
        _, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return thresh

    @staticmethod
    def _apply_deskew(img: Any, cv: Any) -> Any:
        """Correct skew-angle of a binarized text image.

        Finds the minimum-area bounding box of foreground pixels
        and rotates the image to align the text horizontally.
        """
        np = _get_numpy()
        if np is None:
            raise RuntimeError("numpy is required for deskew preprocessing")

        # Invert so text is white on black for contour detection.
        inverted = cv.bitwise_not(img)
        coords = np.column_stack(np.where(inverted > 0))

        if coords.shape[0] < 100:  # too few foreground pixels
            return img

        angle = cv.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:  # already straight
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_REPLICATE,
        )
        logger.debug("Deskewed by %.2f°", angle)
        return rotated

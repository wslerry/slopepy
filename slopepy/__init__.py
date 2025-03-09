"""Slopy - GPU-Accelerated DEM Analysis Toolkit

A package for Digital Elevation Model analysis including hillshade,
slope analysis, geomorphology, and hydrology calculations using GPU acceleration.
"""

__version__ = "0.1.0"
from .classification import GPUSuitabilityClassifier
from .contour import GPUContourGenerator
from .hillshade import HillshadeCalculatorGPU
from .smoothing import FeaturePreservingSmoothing

__all__ = ['GPUSuitabilityClassifier', 
           'GPUContourGenerator', ''
           'HillshadeCalculatorGPU', 
           'FeaturePreservingSmoothing']
#!/usr/bin/env python3
"""
Test runner script for Market Analyzer project.
"""

import os
import sys
import unittest

if __name__ == '__main__':
    # Add the current directory to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Discover and run all tests in the 'tests' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    # Run the tests with a text test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with a non-zero code if there were failures or errors
    sys.exit(not result.wasSuccessful()) 
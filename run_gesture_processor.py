#!/usr/bin/env python3
"""
Gesture Log Processor - Command Line Interface

This script provides a command-line interface to process CSV files with MediaPipe landmarks
and generate gesture logs.

Usage:
    python run_gesture_processor.py --input input.csv --output output_directory [options]

Example:
    python run_gesture_processor.py --input recordings/session1.csv --output results
"""

import os
import sys
import argparse
from gesture_log_processor import preprocess_csv, process_csv

# Add parent directory to path to import from config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Default values in case config import fails
DEFAULT_WINDOW_SIZE = 10
DEFAULT_STEP_SIZE = 1
DEFAULT_EXCLUDED_LANDMARKS = None

# Try to import config values
try:
	from config import (
		SLIDING_WINDOW_STEP_SIZE,
		SLIDING_WINDOW_SIZE,
		EXCLUDED_LANDMARKS
	)
	
	print("Successfully imported settings from config file")
except ImportError:
	print("Warning: Could not import from config file, using default values")
	SLIDING_WINDOW_STEP_SIZE = DEFAULT_STEP_SIZE
	SLIDING_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
	EXCLUDED_LANDMARKS = DEFAULT_EXCLUDED_LANDMARKS


def main():
	"""Command-line interface for the gesture log processor."""
	parser = argparse.ArgumentParser(
		description="Process CSV files with MediaPipe landmark data to generate gesture logs.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  Process a CSV file with default settings from config:
    python run_gesture_processor.py --input data.csv --output results

  Override config settings:
    python run_gesture_processor.py --input data.csv --output results --window 15 --step 2

  Specify landmarks to exclude:
    python run_gesture_processor.py --input data.csv --output results --exclude nose left_eye right_eye
"""
	)
	
	parser.add_argument("--input", required=True,
	                    help="Path to input CSV file with landmark data")
	parser.add_argument("--output", default="./output",
	                    help="Directory to save output files")
	parser.add_argument("--window", type=int, default=None,
	                    help=f"Size of sliding window for gesture detection (default from config: {SLIDING_WINDOW_SIZE})")
	parser.add_argument("--step", type=int, default=None,
	                    help=f"Step size for sliding window (default from config: {SLIDING_WINDOW_STEP_SIZE})")
	parser.add_argument("--exclude", nargs='+', default=None,
	                    help=f"List of landmarks to exclude (default from config: {EXCLUDED_LANDMARKS})")
	parser.add_argument("--verbose", action="store_true",
	                    help="Enable verbose output")
	
	args = parser.parse_args()
	
	# Use config values as defaults if not specified in command line
	window_size = args.window if args.window is not None else SLIDING_WINDOW_SIZE
	step_size = args.step if args.step is not None else SLIDING_WINDOW_STEP_SIZE
	excluded_landmarks = args.exclude if args.exclude is not None else EXCLUDED_LANDMARKS
	
	# Check if input file exists
	if not os.path.exists(args.input):
		print(f"Error: Input file {args.input} not found")
		sys.exit(1)
	
	# Print summary of settings
	print("\nGesture Log Processor")
	print("====================")
	print(f"Input file:         {args.input}")
	print(f"Output directory:   {args.output}")
	print(f"Window size:        {window_size}" + (" (from config)" if args.window is None else " (from command line)"))
	print(f"Step size:          {step_size}" + (" (from config)" if args.step is None else " (from command line)"))
	print(f"Excluded landmarks: {excluded_landmarks or 'None'}" +
	      (" (from config)" if args.exclude is None else " (from command line)"))
	print(f"Verbose mode:       {'Enabled' if args.verbose else 'Disabled'}")
	print("\nStarting processing...")
	
	try:
		# Preprocess the data to filter out excluded landmarks
		augmented_file = preprocess_csv(args.input, args.output, excluded_landmarks)
		
		# Process the augmented file to generate gesture logs
		gesture_log, performance_results = process_csv(
			augmented_file,
			args.output,
			window_size=window_size,
			step_size=step_size
		)
		
		print("\n✅ Processing completed successfully!")
		print(f"📄 Gesture log:         {gesture_log}")
		print(f"📄 Performance results: {performance_results}")
		print("\nYou can now use these files for gesture analysis or performance evaluation.")
	
	except Exception as e:
		print(f"\n❌ Error during processing: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
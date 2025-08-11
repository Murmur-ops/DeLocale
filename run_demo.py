#!/usr/bin/env python3
"""
Main demonstration script for sensor network localization
"""

import os
import sys
from examples.simple_example import run_simple_example
from examples.mps_vs_admm import run_comparison

def main():
    print("=" * 70)
    print(" SENSOR NETWORK LOCALIZATION DEMONSTRATION")
    print(" MPS and ADMM Algorithms Implementation")
    print("=" * 70)
    
    while True:
        print("\nSelect an option:")
        print("1. Run simple localization example")
        print("2. Compare MPS vs ADMM algorithms")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n" + "-" * 50)
            print("Running Simple Example...")
            print("-" * 50)
            run_simple_example()
            
        elif choice == '2':
            print("\n" + "-" * 50)
            print("Running Algorithm Comparison...")
            print("-" * 50)
            run_comparison()
            
        elif choice == '3':
            print("\nThank you for using the sensor localization demo!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
"""
Test script to verify all components are working correctly.
"""
import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn')
    ]
    
    missing = []
    
    for name, import_path in required_packages:
        try:
            __import__(import_path)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def test_files():
    """Check that all required scripts exist."""
    print("\nChecking script files...")
    
    required_files = [
        'fetch_funding_data.py',
        'analyze_extreme_funding.py',
        'backtest_strategy.py',
        'run_all.py',
        'requirements.txt',
        'README.md'
    ]
    
    import os
    missing = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠ Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All script files present!")
        return True


def test_api_connection():
    """Test connection to Hyperliquid MAINNET API."""
    print("\nTesting Hyperliquid MAINNET API connection...")
    
    try:
        import requests
        
        url = "https://api.hyperliquid.xyz/info"
        payload = {"type": "meta"}
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'universe' in data:
            num_pairs = len(data['universe'])
            print(f"  ✓ API connection successful!")
            print(f"  ✓ Found {num_pairs} trading pairs")
            return True
        else:
            print("  ✗ Unexpected API response")
            return False
            
    except Exception as e:
        print(f"  ✗ API connection failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("EXTREME FUNDING RATE PROJECT - VERIFICATION TEST")
    print("="*80)
    
    results = []
    
    # Test imports
    results.append(("Dependencies", test_imports()))
    
    # Test files
    results.append(("Files", test_files()))
    
    # Test API
    results.append(("API", test_api_connection()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the analysis.")
        print("\nNext step: python run_all.py")
    else:
        print("\n⚠ Some tests failed. Please resolve issues before proceeding.")
    
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

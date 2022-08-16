# Comparisons on linear unit-tests

## To reproduce the results, run
For experiments with single environment
```bash
python scripts/sweep.py --n_envs 1 --output_dir test_results/1envs --d 0
```

For experiments with three environments
```bash
python scripts/sweep.py --n_envs 3 --output_dir test_results/3envs --d 0
```

For experiments with six environments
```bash
python scripts/sweep.py --n_envs 6 --output_dir test_results/6envs --d 0
```

## And then analyze by run
`test_peak` means the number of test sample used for validation.
It can take values in `0, 20, 100, 500`.
`0` means using data from train distribution for validation.

```bash
python scripts/collect_results.py test_results/1envs --test_peak 20
python scripts/collect_results.py test_results/3envs --test_peak 20
python scripts/collect_results.py test_results/6envs --test_peak 20
```

## For linearly seperable test
```bash
python scripts/high_dim_test.py
```

Our code is based on the [InvarianceUnitTests](https://github.com/facebookresearch/InvarianceUnitTests) suite and [IB-IRM](https://github.com/ahujak/IB-IRM) code.
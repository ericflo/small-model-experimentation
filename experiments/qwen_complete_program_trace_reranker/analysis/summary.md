# Complete-Program Trace Reranker Analysis Summary

Best verifier epoch: 14

| split | base | soft trace | learned | pair rerank | oracle | learned gap recovered | avg candidates | answer positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation mixed L6 | 59.4% | 43.0% | 60.2% | n/a | 98.4% | 2.0% | 111.0 | 30.4 |
| fresh standard L6 | 68.8% | 46.1% | 70.3% | n/a | 98.4% | 5.3% | 111.0 | 30.3 |
| fresh paraphrase L6 | 57.8% | 35.2% | 60.2% | n/a | 100.0% | 5.6% | 111.0 | 27.9 |
| fresh paired L6 | 57.3% | 43.2% | 55.2% | 54.7% | 98.4% | -5.1% | 111.0 | 30.2 |
| hard standard L8 | 50.0% | 32.0% | 46.9% | n/a | 98.4% | -6.5% | 179.0 | 44.7 |
| hard paraphrase L8 | 38.3% | 27.3% | 35.9% | n/a | 96.9% | -4.0% | 179.0 | 43.2 |
| harder standard L10 | 40.6% | 23.4% | 35.2% | n/a | 98.4% | -9.5% | 263.0 | 47.6 |
| harder paraphrase L10 | 17.2% | 23.4% | 14.1% | n/a | 91.4% | -4.2% | 263.0 | 36.2 |
| arithmetic L6 | 31.2% | 15.6% | 34.4% | n/a | 96.9% | 4.8% | 111.0 | 2.5 |
| calendar L6 | 37.5% | 28.1% | 40.6% | n/a | 100.0% | 5.0% | 111.0 | 17.0 |
| unit L6 | 40.6% | 28.1% | 43.8% | n/a | 96.9% | 5.6% | 111.0 | 3.5 |
| list L6 | 81.2% | 56.2% | 84.4% | n/a | 100.0% | 16.7% | 111.0 | 54.7 |
| boolean L6 | 87.5% | 84.4% | 78.1% | n/a | 100.0% | -75.0% | 111.0 | 74.4 |
| lookup L6 | 65.6% | 40.6% | 65.6% | n/a | 100.0% | 0.0% | 111.0 | 28.1 |

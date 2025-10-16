See subgrid_cache.h / computeSubgridCache

This is data represents 8x8 patterns with only bits in one of the four 7x7 subgrids
that have simulations lasting 180 or longer generations.

We use this data with --subgrid-cache-file to speed up searching.

To use this data:
```
cat 7x7subgrid-cache.json.gz.part_* | gzip -dc > 7x7subgrid-cache.json
```

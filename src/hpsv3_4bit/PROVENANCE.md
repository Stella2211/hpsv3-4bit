# Reusable runtime provenance

This package adapts the inference implementations and reward model classes
from the sibling HPSv3 projects in this repository. HPSv3 source is MIT and
is identified in `hpsv3/model.py`. The HPSv3++ reward model is copied from the
pinned `PlantPotatoOnMoon/HPSv3-PlusPlus` submodule at commit
`6a095f68ee98330bf22365f872ed609bd44a216f`. That upstream tree does not carry
a license file; permission to redistribute the copied HPSv3++ implementation
is therefore unresolved. This file is retained in wheels and vendor snapshots
so the limitation is visible to reviewers and distributors.

# Reusable runtime provenance

This package is the maintained inference implementation for both CLIs and host
integrations. The sibling projects retain only conversion-specific model
construction. HPSv3 source is MIT and is identified in `hpsv3/model.py`.

HPSv3++ compatibility is provided through a thin adapter that imports the
original implementation from the pinned external source revision
`6a095f68ee98330bf22365f872ed609bd44a216f`. The upstream tree has no license
file, so permission to use or redistribute that code remains unresolved. The
runtime distribution does not copy the upstream checkout or its source bodies;
the marker `LicenseRef-HPSv3PlusPlus-Permission-Unconfirmed` remains a release
blocker until permission or a licensed replacement is documented.

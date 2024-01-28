from shapely.ops import linemerge
from shapely import LineString, MultiLineString


def merge_linestrings(linestrings, to_multilinestring=False):
    """
    Merges a list of LineString objects into a MultiLineString or a single LineString
    using Shapely's linemerge function.
    
    Args:
    - linestrings (list): A list of LineString objects.
    - to_multilinestring (bool): If True, force output to be a MultiLineString.

    Returns:
    - LineString/MultiLineString: The merged LineString or MultiLineString object.
    """

    valid_linestrings = [ls for ls in linestrings if not ls.is_empty]

    if not valid_linestrings:
        return LineString()

    merged = linemerge(valid_linestrings)
    if to_multilinestring and not isinstance(merged, MultiLineString):
        return MultiLineString([merged])
    
    return merged


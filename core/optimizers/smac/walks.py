from ConfigSpace import Configuration


class RecordedWalk(list):
    """`LocalSearch._search` replaces list entries as it climbs, so
    `RecordedWalk` records each entry's whole path with SMAC unchanged.
    """

    def __init__(self, start_points, sink):
        super().__init__(start_points)
        self.paths = [[point] for point in start_points]
        sink.extend(self.paths)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        path = self.paths[index]
        if path[-1] != value:
            path.append(value)


def record_walks(smac) -> list:
    """`record_walks` collects every later local-search path and pins the
    search to one process, since worker paths never return."""
    selector = getattr(smac.intensifier, "_config_selector", None)
    maximizer = getattr(selector, "_acquisition_maximizer", None)
    local = getattr(maximizer, "_local_search", maximizer)
    paths: list = []
    if local is None or not hasattr(local, "_search"):
        return paths
    search = local._search
    local.n_jobs = 1

    def recording(start_points):
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        return search(RecordedWalk(start_points, paths))

    local._search = recording
    return paths


def walks_to(asked, paths) -> list:
    """`walks_to` finds, for each asked configuration, the unused local-search
    path that ended at it, or None when no climb reached it."""
    used: set = set()
    out = []
    for info in asked:
        match = None
        for index, path in enumerate(paths):
            if index not in used and len(path) and path[-1] == info.config:
                match = index
                break
        if match is None:
            out.append(None)
            continue
        used.add(match)
        out.append([dict(point) for point in paths[match]])
    return out

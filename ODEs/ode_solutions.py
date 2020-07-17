from ODEs.xppcall import xpprun


def solve_from_file(filepath):
    npa, variables = xpprun(filepath, clean_after=True)

    i_st = 100
    i_end = 203

    times = npa[i_st:i_end,0]
    npa = npa[i_st:i_end,:]

    series = lambda name : npa[:, 1+variables.index(name)]
    variables = [var.upper() for var in variables]
    data = {var : series(var) for var in variables}
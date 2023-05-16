import dolfin as df
import numpy as np

from fenicsstorage import FenicsStorage


def function_interpolator(data, times):
    dt = times[1:] - times[:-1]
    dudt = [(d1 - d0) / dti for d0, d1, dti in zip(data[:-1], data[1:], dt)]

    def call(t):
        if t <= times[0]:
            return data[0]
        if t >= times[-1]:
            return data[-1]
        bin = np.digitize(t, times) - 1
        return data[bin] + dudt[bin] * (t - times[bin])

    return call


def vectordata_interpolator(data, times):
    dt = times[1:] - times[:-1]
    dudt = [
        (d1.vector() - d0.vector()) / dti
        for d0, d1, dti in zip(data[:-1], data[1:], dt)
    ]

    def call(t):
        if t <= times[0]:
            return data[0].vector()
        if t >= times[-1]:
            return data[-1].vector()
        bin = np.digitize(t, times) - 1
        return data[bin].vector() + dudt[bin] * (t - times[bin])

    return call


def interpolate_from_file(filepath, name, t):
    store = FenicsStorage(filepath, "r")
    tvec = store.read_timevector(name)
    bin = np.digitize(t, tvec) - 1
    C = [store.read_function(name, idx=i) for i in range(tvec.size)[bin:bin+2]]
    interpolator = vectordata_interpolator(C, tvec[bin:bin+2])
    u = df.Function(C[0].function_space())
    u.vector()[:] = interpolator(t)
    store.close()
    return u

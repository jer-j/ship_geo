"""CSDL value and first-derivative checks for the coefficient operations."""

import importlib.util
from pathlib import Path
import csdl_alpha as csdl
import numpy as np
from numpy.testing import assert_allclose

path=Path(__file__).parents[1]/'lsdo_geo/core/ship_geometry/stem_integration.py'
spec=importlib.util.spec_from_file_location('stem_integration',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_stem_boundary_and_zero_support():
    recorder=csdl.Recorder(inline=True);recorder.start()
    P=csdl.Variable(value=np.random.default_rng(3).normal(size=(6,6,3)))
    target=csdl.Variable(value=np.zeros((6,3)))
    weight=csdl.Variable(value=np.array([1.,1.,1.,0.,0.,0.]))
    result=module.integrate_stem(P,target,weight)
    assert_allclose(result.value[0],target.value)
    assert_allclose(result.value[3:],P.value[3:])
    recorder.stop()


def test_projection_total_derivative():
    recorder=csdl.Recorder(inline=True);recorder.start()
    P=csdl.Variable(value=np.zeros((6,6,3)))
    weight=csdl.Variable(value=np.array([1.,1.,1.,0.,0.,0.]))
    projection=csdl.Variable(value=1.2);drop=csdl.Variable(value=.7)
    result=module.project_sonar_nose(P,weight,weight,projection,drop)
    assert_allclose(result.value[0,0],[-1.2,0.,-.7])
    derivative=csdl.derivative(result,projection)
    expected=np.zeros((6,6,3));expected[:,:,0]=-np.outer(weight.value,weight.value)
    assert_allclose(derivative.value.ravel(),expected.ravel(),atol=1e-12)
    recorder.stop()

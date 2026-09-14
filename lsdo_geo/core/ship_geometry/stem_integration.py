"""CSDL coefficient operations for integrated stems and projecting sonar noses.

The caller supplies compatible spline spaces. The longitudinal blend must
equal one at the stem and have a zero two-jet at its aft support boundary.
The stem correction must have a zero transverse two-jet at the waterline.
These are coefficient operations, not a new nonlinear geometry solver.
"""

import csdl_alpha as csdl


def integrate_stem(control_net, stem_coefficients, blend_coefficients):
    """Replace the forward boundary while retaining the existing hull surface.

    Parameters
    ----------
    control_net : csdl.Variable
        Compatible tensor-product coefficients with shape (nu, nv, 3).
    stem_coefficients : csdl.Variable
        Target centerplane stem in the same transverse space, shape (nv, 3).
    blend_coefficients : csdl.Variable
        Scalar compact-support blend in the longitudinal space, shape (nu,).

    Returns
    -------
    csdl.Variable
        Coefficients of the integrated surface. With the stated two-jet
        conditions, waterline continuity and aft continuity are preserved.
    """
    nu,nv,dimension=control_net.shape
    if dimension!=3 or stem_coefficients.shape!=(nv,3) or blend_coefficients.shape!=(nu,):
        raise ValueError('Incompatible stem, blend, or control-net shapes.')
    delta=stem_coefficients-control_net[0,:,:].reshape((nv,3))
    return control_net+csdl.expand(blend_coefficients,(nu,nv,3),'i->ijk')*csdl.expand(delta,(nu,nv,3),'jk->ijk')


def project_sonar_nose(control_net, longitudinal_profile, transverse_profile,
                       forward_projection, nose_drop):
    """Add a centerplane nose with differentiable forward and vertical offsets.

    The profiles are B-spline coefficients, not samples. Their tensor product
    must have zero transverse two-jet at the retained-hull attachment. Positive
    projection points forward when the longitudinal coordinate increases aft;
    positive drop points downward. A zero centerplane breadth remains zero.
    """
    nu,nv,_=control_net.shape
    weight=csdl.expand(longitudinal_profile,(nu,nv),'i->ij')*csdl.expand(transverse_profile,(nu,nv),'j->ij')
    x=control_net[:,:,0].reshape((nu,nv))-forward_projection*weight
    z=control_net[:,:,2].reshape((nu,nv))-nose_drop*weight
    result=control_net.set(csdl.slice[:,:,0],x)
    return result.set(csdl.slice[:,:,2],z)

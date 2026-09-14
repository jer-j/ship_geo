# Integrated stem and forward sonar controls

The weather-deck curve should be a boundary of the whole topside surface.
A separate foredeck connector requires its own tangent and curvature joins.
The previously distributed prototype had an 8.30195 degree tangent jump at
the forward perpendicular. The integrated construction removes that seam.

For compatible tensor-product spaces, use

\[
S_{new}(u,v)=S(u,v)+w(u)[C_{stem}(v)-S(0,v)].
\]

The scalar blend equals one at the stem and has zero value and first two
derivatives at its aft support boundary. The stem correction has zero first
two transverse derivatives and zero value at the waterline. These conditions
retain the waterline two-jet and give second-order contact at the aft end of
the integrated region. The side deck is one spline boundary, not a Coons edge.

`integrate_stem` performs this coefficient update using native CSDL operations.
Compatible spaces and the stated profile constraints are caller responsibilities.
There is no SciPy solve or JAX custom callback in this operation.

`project_sonar_nose` provides independent forward and downward amplitudes:

\[
\Delta x=-p\,w_n(u)g(v),\qquad
\Delta z=-h\,w_n(u)g(v),\qquad\Delta y=0.
\]

The profiles have zero two-jets at the retained-hull attachment. Projection
and drop are displacements from the supplied geometry, respectively forward
and downward where the profile product is one. Preserve the centerplane boundary and
verify surface regularity when selecting amplitudes.

Focused tests check boundary replacement, unchanged support, nose coordinates
and the CSDL projection derivative. The whole-deck extension also tests a
shared breadth/sheer solve, implicit derivatives, C2 knots, and compatible
deck/stem composition. All five focused tests now pass on Python 3.10 with
the repository-pinned dependencies. See [whole weather deck](whole_weather_deck.md)
for the native reference fit. Full hull assembly remains a separate step;
these operators do not establish an end-to-end calibrated reconstruction.

The reference is the existing checksum-pinned DTMB 5415 IGES in
`lsdo_geo/validation/dtmb_5415.py`. A side edge can be smooth while the closed
deck perimeter retains intentional corners at the stem and transom. Neither
nominal dimension matching nor surface continuity establishes a geometry fit.

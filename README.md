# Numerical Scheme for the Kinetic Fokker–Planck Equation (1D)

This project develops and implements a **numerical scheme** for the **kinetic Fokker–Planck equation** in one spatial and one velocity dimension. The equation has the general form

$$ \partial_t f=\mathcal{L}f:=-v\cdot\nabla_x f+\nabla_x V\cdot \nabla_v f+\nabla_v\cdot \left(\mathcal{M}\nabla_v\left(\frac{f}{\mathcal{M}}\right)\right), $$

where:
- The **transport operator** corresponds to a **classical Hamiltonian vector field** with external potential $V(x)$,
- The **collision operator** is a **Fokker–Planck operator in velocity**.

---

## Domain and Boundary Conditions

- The computation is performed on a **very large domain** in both **space** $x$ and **velocity** $v$.
- **Dirichlet boundary conditions** are imposed at the boundaries of the computational box.

---

## Numerical Method

The scheme is based on an **operator splitting strategy**:

1. **Transport step (explicit):**
   - The transport part is discretized with a **finite difference scheme**.
   - The **CFL condition** governs the choice of the time step to ensure stability.
   
2. **Collision step (implicit):**
   - The collision operator is treated with the **Chang–Cooper scheme** in velocity.
   - This ensures **positivity preservation** and a correct **long-time equilibrium**.
   - The scheme is solved **implicitly**, making it **unconditionally stable** in this step.

---

## Features

- Splitting method combining explicit (transport) and implicit (collision) discretizations.
- **Positivity-preserving** and **conservative** treatment of the collision term.
- Suitable for exploring **long-time dynamics** and **equilibrium states** $G$.

---

## Possible Future Directions

- Extension to higher-dimensional problems (2D).

---
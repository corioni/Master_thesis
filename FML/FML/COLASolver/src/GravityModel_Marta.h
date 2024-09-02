#ifndef GRAVITYMODEL_MARTA_HEADER
#define GRAVITYMODEL_MARTA_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include <iostream>

using Spline = FML::INTERPOLATION::SPLINE::Spline;

template <int NDIM>
class GravityModelMarta : public GravityModel<NDIM> {
  private:
    double mu0;
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    GravityModelMarta() : GravityModel<NDIM>("Geff(a)") {}
    GravityModelMarta(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "Marta") {}

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi = norm_poisson_equation * delta
    //========================================================================
    void compute_force(double a,
                       [[maybe_unused]] double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Computes gravitational force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a * GeffOverG(a);
        
        if (this->force_use_finite_difference_force) {
          // Use a by default a 4 point formula (using phi(i+/-2), phi(i+/-1) to compute DPhi)
          // This requires 2 boundary cells (stencil_order=2,4,6 implemented so far)
          const int stencil_order = this->force_finite_difference_stencil_order;
          const int nboundary_cells = stencil_order/2;

          FFTWGrid<NDIM> potential_real;
          FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(density_fourier,
              potential_real,
              norm_poisson_equation,
              nboundary_cells);

          FML::NBODY::compute_force_from_potential_real<NDIM>(potential_real,
              force_real,
              density_assignment_method_used,
              stencil_order);

        } else {
          // Computes gravitational force using fourier-methods
          FML::NBODY::compute_force_from_density_fourier<NDIM>(
              density_fourier, force_real, density_assignment_method_used, norm_poisson_equation);
        }
    }

    //========================================================================
    // In JBD GeffOverG = 1/phi. The value at 1/phi(a=1) is the parameter
    // cosmology_JBD_GeffG_today
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0) const override { 
      return 1.0 + mu0 * this->cosmo->get_OmegaLambda(a) / this->cosmo->get_OmegaLambda();
    }

    //========================================================================
    // Read parameters
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
    
        mu0 = param.get<double>("gravity_model_marta_mu0");
        if(FML::ThisTask == 0)
          std::cout << "My mu0 = " << mu0 << std::endl;

        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }
    
    //========================================================================
    // Show some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "GeffG(a=1.00) = " << GeffOverG(1.00) << "\n";
            std::cout << "GeffG(a=0.66) = " << GeffOverG(0.66) << "\n";
            std::cout << "GeffG(a=0.50) = " << GeffOverG(0.50) << "\n";
            std::cout << "GeffG(a=0.33) = " << GeffOverG(0.33) << "\n";
            std::cout << "GeffG(a=0.25) = " << GeffOverG(0.25) << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif

using Parameters
using ForwardDiff
using NLsolve
using DifferentialEquations
using FoodWebs
using PyPlot

pygui(true)


@with_kw mutable struct Par
    r = 2.0
    K = 3.0
    a = 1.1
    h = 0.8
    e = 0.7
    m = 0.4
    σ = 0.1
end

# Standard inplace version
function roz_mac!(du, u, p, t,)
    @unpack r, K, a, h, e, m = p
    R, C = u
    du[1] = r * R * (1 - R / K) - a * R * C / (1 + a * h * R)
    du[2] = e * a * R * C / (1 + a * h * R) - m * C
    return
end

# We need this wrapper to make it a "function" for the AutoDiff library
function roz_mac(u, par)
    du = similar(u)
    roz_mac!(du, u, par, 0.0)
    return du
end

# λ_stability is from FoodWebs.jl -- it is a trivial function being:
# maximum(real.(eigvals(M))), where eigvals is from the standard library LInearAlgebra
calc_λ1(eq, par) = λ_stability(ForwardDiff.jacobian(eq -> roz_mac(eq, par), eq))
calc_ν(eq, par) = ν_stability(ForwardDiff.jacobian(eq -> roz_mac(eq, par), eq))


tspan = (0.0, 1000.0)
u0 = [2.0, 1.0]

par = Par()
prob = ODEProblem(roz_mac!, u0, tspan, par)

sol = solve(prob)

eq = nlsolve((du, u) -> roz_mac!(du, u, par, 0.0), sol.u[end]).zero

let
    nsamp = 100
    avals = range(0.4, 1.2, length = nsamp)
    stab = fill(NaN, 2, nsamp)

    for (i, a) in enumerate(avals)
        par.a = a
        prob = ODEProblem(roz_mac!, u0, tspan, par)
        sol = solve(prob)
        eq = nlsolve((du, u) -> roz_mac!(du, u, par, 0.0), sol.u[end]).zero
        stab[1, i] = calc_λ1(eq, par)
        # This is using the equilibrium approximation, but maybe it is worth numerically
        # calculating the reactivity in the excitable case to see if it acts more "cycle like"
        # That is do we see an early onset of the reactivity increasing if we look at the local
        # distribution of deviations around the complex eigenvalues
        stab[2, i] = calc_ν(eq, par)
    end

    figure()
    subplot(311)
    plot(avals, stab[1, :])
    xlabel("a")
    ylabel(L"\lambda_1")

    subplot(312)
    plot(avals, stab[2, :])
    xlabel("a")
    ylabel(L"\nu")

    subplot(313)
    plot(stab[1, :], stab[2, :], "o")
    xlabel(L"\lambda_1")
    ylabel(L"\nu")
    tight_layout()
end

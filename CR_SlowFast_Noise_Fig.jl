using Parameters
using ForwardDiff
using NLsolve
using DifferentialEquations
using Statistics
using FoodWebs
using Noise
using PyPlot

pygui(true)

@with_kw mutable struct Par
# we note that maximum consumer growth rate (r) is driven by a*e relative to m or (ae/M)
# K than fuels this max growth rate s.t in high K max r (max ae/m) is more "realized"
    r = 2.0
    K = 3.0
    a = 1.1
    h = 0.8
    e = 0.7
    m = 0.4
    σ = 0.1
    noise=0.01
end

# Standard inplace version
function roz_mac!(du, u, p, t,)
    @unpack r, K, a, h, e, m = p
    R, C = u
    du[1] = r * R * (1 - R / K) - a * R * C / (1 + a * h * R)
    du[2] = e * a * R * C / (1 + a * h * R) - m * C
    return
end

# noise for time series runs
function stoch_rozmac!(du, u, p2, t)
    @unpack  noise = p2

    du[1] = noise * u[1]
    du[2] = noise * u[2]
    return du 
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
    subplot(211)
    plot(avals, stab[1, :])
    xlabel("a",fontsize=16,fontweight=:bold)
    ylabel(L"\lambda_1",fontsize=16,fontweight=:bold)

    subplot(212)
    plot(avals, stab[2, :])
    xlabel("a",fontsize=16,fontweight=:bold)
    ylabel(L"\nu",fontsize=16,fontweight=:bold)

    tight_layout()

# reactivity
#    plot(avals, stab[2, :])
#    xlabel("a")
#    ylabel(L"\nu")
end

# stochastic C-R time series -- just to look at timeseries specifically whenever we want 

    u0 = [0.5, 0.5]
    t_span = (0.0, 1000.0)
    p = Par(a=0.35,noise =0.05)

    prob_stoch = SDEProblem(roz_mac!, stoch_rozmac!, u0, t_span, p)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    adapt_rozmacts = figure()
    plot(sol_stoch.t[991:end], sol_stoch[2, 991:end])
    xlabel("time")
    ylabel("Density")
    legend(["C"])
    return adapt_rozmacts

## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
ts = range(0, 1000, length = 1000)
## Then we just need to *call* the solution object
grid_sol = sol_stoch(ts)
grid_sol.t
grid_sol.u
plot(1:1000, grid_sol[2,1:1000])
xlim(1, 1000)

# now loop over increasing ae/m -- by increasing a 
# 
# loop over i but really subbing in new a parmaeters as amaxs

amaxs = 0.35:0.01:1.50
maxhold = fill(0.0,length(amaxs),2)
stdhold = fill(0.0,length(amaxs),1)
meanhold = fill(0.0,length(amaxs),1)
cvhold = fill(0.0,length(amaxs),1)
print(amaxs[81])


## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
ts = range(0, 1000, length = 1000)
tspan = (0.0, 1000.0)

for i=1:length(amaxs)
    p = Par(a=amaxs[i], noise=0.04)
    print(p)
       u0 = [0.5928383950718732, 0.6974002406728528]
       prob_stoch = SDEProblem(roz_mac!, stoch_rozmac!, u0, t_span, p)
       sol_stoch = solve(prob_stoch, reltol = 1e-15)

## Then we just need to *call* the solution object
    grid_sol = sol_stoch(ts)
    grid_sol.t
    grid_sol.u
    maxhold[i,1]=amaxs[i]
    maxhold[i,2]=maximum(grid_sol[2,900:1000])
    stdhold[i]=std(grid_sol[2,900:1000])
    meanhold[i]=mean(grid_sol[2,900:1000])
    cvhold[i] = stdhold[i]/meanhold[i]
end

# fig c
plot(maxhold[1:length(amaxs),1],stdhold[1:length(amaxs)])
xlabel("attack rate",fontsize=16,fontweight=:bold)
ylabel("Standard Devation (C)",fontsize=16,fontweight=:bold)
# fic d
plot(maxhold[1:length(amaxs),1],meanhold[1:length(amaxs)])
xlabel("attack rate",fontsize=16,fontweight=:bold)
ylabel("Mean (C)",fontsize=16,fontweight=:bold)
#fig e
plot(maxhold[1:length(amaxs),1],cvhold[1:length(amaxs)])
xlabel("atack rate",fontsize=16,fontweight=:bold)
ylabel("CV (C)",fontsize=16,fontweight=:bold)

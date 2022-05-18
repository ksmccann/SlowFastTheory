using Parameters
using ForwardDiff
using LinearAlgebra
using PyPlot
using DifferentialEquations
using NLsolve
using Statistics
using RecursiveArrayTools
using Noise
using Distributed
using StatsBase


pygui(true)

## Here we will illustrate the case of a generalist predator that feeds on fast and slow channel.
## Parameters are categorized by macrohabitat but coud be named fast slow more generally  -> parameters with "_litt" or slow indicate littoral macrohabitat values and those with "_pel" or fast indicate pelagic macrohabitat values  


@with_kw mutable struct AdaptPar     
    α_pel = 0.0   ##competitive influence of pelagic resource on littoral resource 
    α_litt = 0.0   ## competitve influence of littoral resource on pelagic resource
    k_litt = 1.0 
    k_pel = 1.0
    e_CR = 0.8
    e_PC = 0.8
    e_PR = 0.8
    m_P = 0.4
    a_PR_litt = 0.00

    # slow ap=.75    
    a_PC_litt= 2.025
    a_PR_pel = 0.00    
    a_PC_pel= 6.75
    h_PC = .3750
    h_PR = 0.80
  
# slow CR
    r_litt = 1.0
    a_CR_litt = 3.250
    m_Cl= 0.70
    h_CRl=0.60

# fast CR
    r_pel = 1.0
    a_CR_pel = 7.50
    m_Cp =0.70
    h_CRp =0.25

# noise
    σ = 6
    noise = 0.03

end


## Omnivory Module with Temp Dependent Attack Rates (a_PC_litt => aPC in littoral zone; a_PC_pel => aPC in pelagic zone)

function adapt_model!(du, u, p, t)
    @unpack r_litt, r_pel, k_litt, k_pel, α_pel, α_litt, e_CR, e_PC, e_PR, a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, h_CRl, h_CRp,h_PC, h_PR, m_Cl, m_Cp,m_P, a_PC_litt,a_PC_pel, σ = p 
    
    
    R_l, R_p, C_l, C_p, P = u
    
    du[1]= r_litt * R_l * (1 - (α_pel * R_p + R_l)/k_litt) - (a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CRl * R_l) - (a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p)
    
    du[2] = r_pel * R_p * (1 - (α_litt * R_l + R_p)/k_pel) - (a_CR_pel * R_p * C_p)/(1 + a_CR_pel * h_CRp * R_p) - (a_PR_pel * R_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p)

    du[3] = (e_CR * a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CRl * R_l) - (a_PC_litt * C_l * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p) - m_Cl * C_l
    
    du[4] = (e_CR * a_CR_pel * R_p * C_p)/(1 + a_PC_pel * h_CRp * R_p) - (a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p) - m_Cp * C_p

du[5] = (e_PR * a_PR_litt * R_l * P + e_PR * a_PR_pel * R_p * P + e_PC * a_PC_litt * C_l * P + e_PC * a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p) - m_P * P

    return 
end

## Adding stochasticity to model using gaussian white noise (SDEproblem)

function stoch_adapt!(du, u, p2, t)
    @unpack  noise = p2

    du[1] = noise * u[1]
    du[2] = noise * u[2]
    du[3] = noise * u[3]
    du[4] = noise * u[4]
    du[5] = noise * u[5]

    return du 
end


## Plotting time series with noise 


    u0 = [0.5, 0.50, 0.30, 0.30, 0.150]
    t_span = (0.0, 10000.0)
    p = AdaptPar(noise =0.0)

    prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, t_span, p)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    
    ## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
    ts2 = range(0, 10000, length = 10000)
    ts = range(0, 10000, length = 10000)
    ## Then we just need to *call* the solution object
  
    grid_sol = sol_stoch(ts)
    grid_sol.t
    grid_sol.u
    plot(grid_sol.u)
    
    plot(grid_sol[5,1:1000])

    xlabel("time")
    ylabel("Density")
    legend(["P"])
    xlim(9000,10000) 

    println(grid_sol.u) 
 
    # for check of ode in more complex, do lower dimensional checks

# amax=1/h, Ro=1/(ah)
# slow CR
r= 1.0
a= 3.50
m= 0.60
h=0.90

# fast CR
r = 1.0
a = 3.250
m =0.70
h =.6
e=0.80
K=1.0


    Rhump = K - K/(a * h)
    Rhold = m/(e * a - m * a * h)
    Chold = r/a*(1-Rhold/K)*(1 + a * h * Rhold)
     

# having seen the time series and checked that the ode works at cases with clear answers we now proceed to 
# do CV calculations over a range of weak/slow litt to symetrically fast/strong both channels 
# We are increasing all the parms of the slow channel until the symmetry point
Fast_mult = 0.0:0.01:1.0


stdhold = fill(0.0,length(Fast_mult),1)
meanhold = fill(0.0,length(Fast_mult),1)
cvhold = fill(0.0,length(Fast_mult),1)

stdhold34 = fill(0.0,length(Fast_mult),1)
meanhold34 = fill(0.0,length(Fast_mult),1)
cvhold34 = fill(0.0,length(Fast_mult),1)

print(Fast_mult[1])

# initial slow parm values
a_CRl_i =  3.25 
a_PCl_i = 2.025
h_CRl_i = .60

for i=1:length(Fast_mult)
u0 = [0.5, 0.50, 0.3, 0.30, 0.30]
t_span = (0.0, 10000.0)

p = AdaptPar(a_CR_litt=a_CRl_i+Fast_mult[i]*(7.50-a_CRl_i), a_PC_litt=a_PCl_i+Fast_mult[i]*(6.5-a_PCl_i), h_CRl=h_CRl_i-Fast_mult[i]*(h_CRl_i-0.250), noise=0.02)
#p = AdaptPar(noise =0.01)

prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, t_span, p)
sol_stoch = solve(prob_stoch, reltol = 1e-15)
print(i)

## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
ts2 = range(0, 10000, length = 10000)
ts = range(0, 10000, length = 10000)
## Then we just need to *call* the solution object at time steps of 1 for time series work
grid_sol = sol_stoch(ts)

# now do CV calcs
    stdhold[i]=std(grid_sol[5,5000:10000])
    meanhold[i]=mean(grid_sol[5,5000:10000])
    cvhold[i] = stdhold[i]/meanhold[i]

    stdhold34[i]=std(grid_sol[4,5000:10000])
    meanhold34[i]=mean(grid_sol[4,5000:10000])
    cvhold34[i] = stdhold34[i]/meanhold[i]


end


print(stdhold)
print(meanhold)
print(cvhold)

plot(stdhold,Linewidth=4)
xlabel("Slow to Fast",fontsize=16,fontweight=:bold)
ylabel("Std Dev",fontsize=16,fontweight=:bold)

 
plot(meanhold,Linewidth=4)
xlabel("Slow to Fast",fontsize=16,fontweight=:bold)
ylabel("Mean",fontsize=16,fontweight=:bold)
legend(["P"],fontsize=16,fontweight=:bold)

plot(cvhold,Linewidth=4)
xlabel("Slow to Fast",fontsize=16,fontweight=:bold)
ylabel("CV",fontsize=16,fontweight=:bold)
legend(["P"],fontsize=16,fontweight=:bold)

plot(cvhold34)
plot(stdhold34)
















using DynamicalSystems
using Statistics
using PyPlot
using Distributions

pygui(true)

# figure 1 Ricker for Slow-Fast
# simple plot of timeseries -- avoiding Bif plot for more readable to general audience

# for later random noise added

f(x, p, t) = x * exp(p[1] * x * (1 - x / p[2])) + p[3] * rand(Normal(0, 1)) 
x0 = .30 # initial condition
p0 = [0.50, 1.0,0.10] # parameters r, K

ds = DiscreteDynamicalSystem(f, x0, p0)
tr = trajectory(ds, 100) # timeseries
print(tr[100])
plot(tr)
ylim(0,2.0)
xlabel("Time",fontsize=16,fontweight=:bold)
ylabel("N, Density",fontsize=16,fontweight=:bold)

#################################################################################################3
# do noise-induced or stochastic dynamics after transistent across r and measure CV, STD, mean
rmaxs = 0.5:0.01:2.80
stdhold = fill(0.0,length(rmaxs),1)
meanhold = fill(0.0,length(rmaxs),1)
cvhold = fill(0.0,length(rmaxs),1)
print(rmaxs[100])

for i=1:length(rmaxs)
x0 = .30 # initial condition
p0 = [rmaxs[i], 1.0,0.10]
println(rmaxs[i])
f(x, p, t) = x * exp(p[1] * x * (1 - x / p[2])) + p[3] * rand(Normal(0, 1)) 
x0 = 0.3
ds = DiscreteDynamicalSystem(f, x0, p0)
tr = trajectory(ds, 1000)
    stdhold[i]=std(tr[500:1000])
    meanhold[i]=mean(tr[500:1000])
    cvhold[i] = stdhold[i]/meanhold[i]
end

print(stdhold)

# fig a

plot(rmaxs[1:length(rmaxs)],stdhold[1:length(rmaxs)])
xlabel("r, max growth rate",fontsize=16,fontweight=:bold)
ylabel("Standard Devation (N)",fontsize=16,fontweight=:bold)
# fic b
plot(rmaxs[1:length(rmaxs)],meanhold[1:length(rmaxs)])
xlabel("r, max growth rate",fontsize=16,fontweight=:bold)
ylabel("Mean (N)",fontsize=16,fontweight=:bold)
#fig c
plot(rmaxs[1:length(rmaxs)],cvhold[1:length(rmaxs)])
xlabel("r, max growth rate",fontsize=16,fontweight=:bold)
ylabel("CV (N)",fontsize=16,fontweight=:bold)


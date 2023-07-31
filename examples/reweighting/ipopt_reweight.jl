
using LinearAlgebra
using NLPModels, NLPModelsIpopt, SparseArrays
using NLPModelsAlgencan

#  cd ~/.julia/packages
# sudo rm -r NLPModelsAlgencan
# ENV["MA57_SOURCE"] = "/home/donboyd5/Documents/hsl/hsl_ma57-5.3.2.tar.gz"
# ENV["MA57_SOURCE"] = ""
# println(ENV["MA57_SOURCE"])
# import Pkg
# Pkg.build("NLPModelsAlgencan")
# Pkg.add("NLPModelsAlgencan")

include("mtprw.jl")
include("ipopt_functions.jl")

hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libcoinhsl.so"

function rwscale(A, b)
  # scale xmat so that mean value is 1.0, and scale rwtargets accordingly
  scale = vec(sum(abs.(A), dims=1)) ./ size(A)[1] 
  # scale = fill(1., k) # unscaled
  A = A ./ scale' 
  b = b ./ scale
  # mean(abs.(xmat), dims=1)
  return A, b
end


## get problem
# 100k, 100, 0.3 is too big for ma57
# 100k, 50, 0.3 is too big for ma57, gives incorrect partitioning scheme before bombing
# 100k, 25, 0.3 too big
h = 100_000
k = 100
pzero = 0.4
tp = mtprw(h, k, pctzero=pzero)
tp.h
tp.wh
tp.xmat
tp.rwtargets
tp.rwtargets_calc

## set up problem for ipopt
A = tp.xmat .* tp.wh
b = tp.rwtargets

# A, b = rwscale(tp.xmat .* tp.wh, tp.rwtargets)  

count(!iszero, A)
count(!iszero, A) / length(A)

lvar = fill(0.25, tp.h)
uvar = fill(4.0, tp.h)
tol = .03
lcon=b .- abs.(b)*tol
ucon=b .+ abs.(b)*tol

mod = modcon(A, b; lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon)
mod
mod.meta.nnzj

# check inputs
tcalcs = (tp.xmat .* tp.wh)' * mod.meta.x0
pdiffs = tcalcs ./ tp.rwtargets .- 1.
quantile(pdiffs)

tcalcs = mod.A' * mod.meta.x0
pdiffs = tcalcs ./ b .- 1.
quantile(pdiffs)

# here are some changes on the new reweight branch

res1 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", linear_solver="mumps", mumps_mem_percent=50)
res1a = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma86")
# res2 = algencan(mod)
fieldnames(typeof(res2))

# do not use ma57 on large problems, will abort (why???)
# use mumps ma86, ma97, or possibly ma77 -- compare to mumps
# if ma77, remember to delete an ma77 files in the working directory if not arlready done, before using it a 2nd (or later) time
res2 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma86")

# safe way to run ma77 - avoids crash -- this is important
# tempdir will store the temp files that ma77 creates; we'll delete it after the run because ma77 does not always clean up
tempdir_path = Base.mktempdir()
cd(tempdir_path)
res2 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma77")
cd("..")  # Change back to the parent directory (original directory)
rm(tempdir_path; recursive=true, force=true)


res2 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma57")
# after ma77, go to the directory it ran from (probably pwd()), and do rm ma77_*

# res2 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma97")
# res2 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma57")


#  300k, 200: mumps 465.145; ma57 bomb, ma97 bomb; ma86 377; ma77 489
# scaled: ma86 321

# mumps 8.1, ma86 10.7, ma97 4.7
# decrease the value of the option "mumps_mem_percent
res1.objective
res2.objective

xsol = res1.solution
quantile(xsol)

tcalcs = (tp.xmat .* tp.wh)' * xsol
pdiffs = tcalcs ./ tp.rwtargets .- 1.
quantile(pdiffs)

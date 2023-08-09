# djb notes
#   https://github.com/PSLmodels/taxdata/blob/master/puf_stage2/dataprep.py
#     sets up what I would call xmat .* wh, transposed, where
#     A1 is positive, A2 is negative (but the same absolute values)
#     r is the fraction of the weight above 1, s is the fraction below
#     r corresponds to A1, s corresponds to A2
#     so 1 + r - s = the ratio of new weight to old weight


using JuMP, NPZ, Tulip
using Printf
using Statistics
using LinearAlgebra

function Solve_func(year, tol)

	println("\nSolving weights for $year ...\n\n")

	# we only solve the weights for years where the targets have changed. If the
	# targets have not changed, we don't write the _input.npz file
	if isfile(string(year, "_input.npz"))
		array = npzread(string(year, "_input.npz"))
	else
		println("Skipping solver for $year \n")
		return nothing
	end

	A1 = array["A1"]
	A2 = array["A2"]
	b = array["b"]

	N = size(A1)[2]

    # scaling: determine a scaling vector with one value per constraint
	#  - the goal is to keep coefficients reasonably near 1.0
	#  - multiply each row of A1 and A2 by its specific scaling constant
	#  - multiply each element of the b target vector by its scaling constant
	#  - current approach: choose scale factors so that the sum of absolute values in each row of
	#    A1 and of A2 will equal the total number of records / 1000; maybe we can improve on this

	scale = (N / 1000.) ./ sum(abs.(A1), dims=2)

    A1s = scale .* A1
	A2s = scale .* A2
	bs = scale .* b

	model = Model(Tulip.Optimizer)
	set_optimizer_attribute(model, "OutputLevel", 1)  # 0=disable output (default), 1=show iterations
	set_optimizer_attribute(model, "IPM_IterationsLimit", 100)  # default 100 seems to be enough

    # djb
    #   N is the number of records, which is the number of COLUMNS in the A matrices
    #   j indexes the records, which are COLUMNS of A1s and A2s
    #   i indexes the constraints
    #   I THINK: A1 is the positive value of wh .* xmat and A2 is the negative

	# r and s must each fall between 0 and the tolerance
	@variable(model, 0 <= r[1:N] <= tol) # djb r is the amount above 1 e.g., 1 + 0.40, goes with A1s
	@variable(model, 0 <= s[1:N] <= tol) # djb s is the amount below 1 e.g., 1 - 0.40, goes with A2s

	@objective(model, Min, sum(r[i] + s[i] for i in 1:N))  # djb would be clearer to use j as the index here

	# Ax = b  - use the scaled matrices and vector; djb looks like equality constraints
	@constraint(model, [i in 1:length(bs)], sum(A1s[i,j] * r[j] + A2s[i,j] * s[j]
		                          for j in 1:N) == bs[i])

	optimize!(model)

	println("Termination status: ", termination_status(model))
	@printf "Objective = %.4f\n" objective_value(model)

	r_vec = value.(r)
	s_vec = value.(s)

	npzwrite(string(year, "_output.npz"), Dict("r" => r_vec, "s" => s_vec))

	println("\n")

	# quick checks on results

	# Did we satisfy constraints?
	rs = r_vec - s_vec
	b_calc = sum(rs' .* A1, dims=2)
	check = vec(b_calc) ./ b

	q = (0, .1, .25, .5, .75, .9, 1)
	println("Quantiles used below: ", q)

	println("\nQuantiles of ratio of calculated targets to intended targets: ")
	println(quantile!(check, q))

	# Are the ratios of new weights to old weights in bounds (within tolerances)?
	x = 1.0 .+ r_vec - s_vec  # note the .+
	println("\nQuantiles of ratio of new weight to initial weight: ")
	println(quantile!(x, q))

end


year_list = [x for x in 2012:2033]
tol_list = [0.40, 0.38, 0.35, 0.33, 0.30,
 	    0.45, 0.45, 0.45, 0.45, 0.45,
	    0.45, 0.45, 0.45, 0.45, 0.45,
	    0.45, 0.45, 0.45, 0.45, 0.45, 
	    0.45, 0.5]

# Run solver function for all years and tolerances (in order)
for i in zip(year_list, tol_list)
	Solve_func(i[1], i[2])
end

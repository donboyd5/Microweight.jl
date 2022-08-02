
# CAUTION: I have not finished this example, and may not do so!

using CSV, DataFrames, Parquet, Statistics
import Microweight as mw

# The ACS is the American Community Survey, a widely used survey of households in the United States,
#  prepared by the U.S. Bureau of the Census.
# In this example we use the person-level microdata for teachers in the 5-year ACS ending in 2019.
# The specific subset is:
#    teachers, which I define as occupation code (occp) in the range of 2300:2330, with
#    age (agep) 18+, and with
#    wages (wagp) >= $10,0000
# I chose teachers because it gives us a good number of observations for each state
# The file has 264,948 rows and 23 columns
#  Column names, other than those I constructed, are those given in the ACS documentation.
#  The weight column and the constructed columns are:
#    pwgtp --person weight, which will be used as "wh" (weight for households) in Microweight
#    stabbr -- state abbreviation, constructed from the ACS FIPS code variable, st
#    married -- boolean (1=married), constructed from the ACS marital status variable, mar
#    female -- boolean (1=female), constructed from the ACS gender variable, sex
#    age_lt40 -- boolean (1=age < 40), constructed from the ACS age variable, agep
#    age_ge60 -- boolean (1=age >= 60), constructed from the ACS age variable, agep

# In this example, we solve a more challenging problem. We use observations from 10 states and
# try try to construct weights that will hit targets as follows
#   we have 4 states
#   and 3 targets per state

## locations
# path = raw"C:\Users\donbo\Documents\R_projects\acs_for_microweight\\"
path = "acs/"

## get a subset of ACS data, and sums by state
datafn = "teachers.parquet"
datapathfn = path * "teachers.parquet"
data_all = DataFrame(read_parquet(datapathfn))
describe(data_all)
names(data_all)
propertynames(data_all)
# f = x -> !(x in (:stabbr, :pwgtp))
data_cols = filter(x -> !(x in (:stabbr, :st, :serialno)), propertynames(data_all))
data_all[!, data_cols] = data_all[:, data_cols] .* 1.0
# now we have a data frame with floats in the right places
# we want the data columns to be float - define these columns so we can convert as needed

## get weighted sums by state
sumspathfn = path * "teachers_state_sums.csv"
stsums = DataFrame(CSV.File(sumspathfn))

# we construct and solve a very easy problem below, where:
#   we use 4 states (s=4)
#   we subset the ACS file to

# geotargets_df -- data frame that will be used to construct matrix of geographic targets
geotargets_df = stsums[stsums_rows, cols]
# convert data_cols to float, as needed
geotargets_df[!, data_cols] = geotargets_df[:, data_cols] .* 1.0

# define info for rows and columns we want to get from data
state_rows = ["CA", "NY", "OR", "TX", "WI"]
state_data = ["CA", "NY", "OR", "TX", "WI", "AL", "KY", "PA"]
data_rows = in.(data_all.stabbr, Ref(state_data))
# data_rows = bits = trues(size(data)[1])
stsums_rows = in.(stsums.stabbr, Ref(state_rows))

cols_all = [:pwgtp, :stabbr, :totnum, :married, :pincp, :intp]
# cols_all = vcat(:pwgtp, cols)
cols = filter(x -> x!=:pwgtp, cols_all)
# data_cols = filter(x -> x!=:stabbr, cols_all) # remove stabbr, which will not be part of our data

# adjust the data so that
data = data_all[data_rows, cols_all]
# data[!, data_cols] = data[:, data_cols] .* 1.0
data

# weights for households (wh), h x 1
# pwgtp is the person weight in the ACS, which will be our "wh" variable (national weight for household)
# scale pwgtp
sum(data.pwgtp)
sum(geotargets_df.totnum)
# ratio = sum(data.pwgtp) / sum(stsums.totnum)
ratio = sum(geotargets_df.totnum) / sum(data.pwgtp)
data[!, "pwgtp"] = data.pwgtp * ratio
sum(data.pwgtp)

wh = data[!, :pwgtp]
wh = reshape(wh, length(wh), 1) # we need wh as an array (i.e., matrix, in this case)
# wh = convert(Array{Float64}, data.pwgtp)

# xmat_df -- data frame that will be used to construct matrix of x characteristics
data_cols = filter(x -> x!=:stabbr, cols)
xmat_df = data[!, cols]
# convert data_cols to float, as needed
xmat_df



sum(wh)
sum(geotargets_df.totnum)

# create matrices, and then create GeoweightProblem
xmat = Matrix(xmat_df[:, data_cols])
geotargets = Matrix(geotargets_df[:, data_cols])

## create GeoweightProblem
size(wh) # h x 1
size(xmat) # h x k
size(geotargets) # s x k
prob = mw.GeoweightProblem(wh, xmat, geotargets)


## solve for state weights two ways - using poisson approach and direct approach
resp = mw.geosolve(prob, approach=:poisson, maxiter=1000)
resd = mw.geosolve(prob, approach=:direct, stopval=1e-8, print_interval=10, maxiter=10_000)

## examine results
# elapsed seconds
resp.eseconds
resd.eseconds

# sum of squared percentage differences of calculated targets vs. desired target values
resp.sspd
resd.sspd

# state weights
resp.whs
resd.whs

# percentiles of percentage differences of sums of stateweights vs national weights
resp.wh_pdqtiles
resd.wh_pdqtiles

# percentiles of percentage differences of calculated targets vs. desired target values
resp.targ_pdqtiles
resd.targ_pdqtiles

# correlation of state weights under poisson approach with state weights under direct approach
Statistics.cor(vec(resp.whs), vec(resd.whs))

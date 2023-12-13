clear all
set more off
// source folder for SOEP core
global SOEP_C38 "C:\Users\bruno\papers\soep\soep38" 


// get variables from SOEP core long: 
// 1. pgemplst: employment status,
// 2. pglabgro: gross labor income,
// 3. pgexpft: full time experience: continuous! 
append using "${SOEP_C38}\pgen.dta", keep(syear hid pid pgemplst pglabgro pgexpft)
merge 1:1 pid hid syear using "${SOEP_C38}\ppathl.dta", keepusing(sex gebjahr) keep(3)


// restrict sample
keep if sex == 1 // only men
drop if syear < 2010
drop if syear > 2020
drop if pgemplst != 1 // only full time
drop if pgexpft < 0
drop if pgexpft > 40
qui summarize pglabgro, detail
keep if inrange(pglabgro, r(p1), r(p99)) // drop top 1% and bottom 1% of earners

// prepare estimation

xtset pid syear
ren pgexpft full_time_exp
gen full_time_exp_2 = full_time_exp^2
ren pglabgro wage 

// estimate
xtreg wage full_time_exp full_time_exp_2 i.syear, fe // parametric

replace full_time_exp = round(full_time_exp)
xtreg wage i.full_time_exp i.syear, fe // non-parametric
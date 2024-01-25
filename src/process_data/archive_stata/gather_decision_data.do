clear all
set more off
// source folder for SOEP core
global SOEP_C38 "C:\Users\bruno\papers\soep\soep38" 
// source folder for SOEP-RV
global SOEP_RV "C:\Users\bruno\papers\soep\soep_rv\"

// get variables from SOEP core long (sex, age, employment status, experience) 
append using "${SOEP_C38}\pgen.dta", keep(syear pid hid pgemplst pgexpft)
merge 1:1 pid hid syear using "${SOEP_C38}\ppathl.dta", keepusing(sex gebjahr rv_id) keep(3)
drop _merge

gen age = syear - gebjahr
drop if rv_id < 0 & age >= 60 // !!kickt 600k von 700k raus!!
replace rv_id = . if rv_id < 0

// and SOEP RV (retirement state)
ren syear JAHR
gen MONAT = 12
merge m:1 rv_id JAHR MONAT using "${SOEP_RV}\vskt\SUF.SOEP-RV.VSKT.2020.var.1-0.dta", keepusing(STATUS_2) keep(1 3)
ren JAHR syear

// labor choice: look only at only full-time employed or not employed (together 87%). retirement choice: from SOEP RV
drop if pgemplst < 1 // 0.01 % invalid responses
drop if pgemplst == 2 // part-time, 4.2%
drop if pgemplst == 3 // education, 3.3%
drop if pgemplst == 4 // irregular work, 3.86%
drop if pgemplst == 6 // special needs, 0.13%


ren pgemplst choice
label val choice
replace choice = 0 if choice == 5
replace choice = 2 if STATUS_2 == "RTB"

// period
gen period = age - 25
drop if period < 0 // nochmal 7k raus

// lagged choice
xtset pid syear
gen lagged_choice = l1.choice
drop if missing(lagged_choice) // nochmal 14k raus

// restrict sample (insgesamt 57k raus)
keep if sex == 1 // only men
drop if syear < 2010 // estimation start
drop if syear > 2020 // estimation finish

// policy_state
gen policy_state = 67
replace policy_state = 67 - 2/12 * (1964 - gebjahr) if gebjahr <= 1964 & gebjahr >= 1958
replace policy_state = 66 - 1/12 * (1958 - gebjahr) if gebjahr <= 1958 & gebjahr >= 1947
replace policy_state = 65 if gebjahr < 1947 

// retirement age id
gen retirement_age_id = .

// experience 
drop if pgexpft < 0
drop if pgexpft > 40 // nochmal 4.5k raus
replace pgexpft = round(pgexpft)
rename pgexpft experience

// keep relevant
keep choice period lagged_choice policy_state retirement_age_id experience
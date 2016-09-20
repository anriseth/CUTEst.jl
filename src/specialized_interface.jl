export usetup, csetup, udimen, udimsh, udimse, uvartype, unames,
    ureport, cdimen, cdimsj, cdimsh, cdimse, cstats, cvartype, cnames,
    creport, connames, pname, probname, varnames, ufn, ugr, uofg, ubandh,
    udh, ush, ueh, ugrdh, ugrsh, ugreh, uhprod, cfn, cofg, cofsg, ccfg,
    clfg, cgr, csgr, ccfsg, ccifg, ccifsg, cgrdh, cdh, csh, cshc, ceh,
    cidh, cish, csgrsh, csgreh, chprod, chcprod, cjprod, uterminate,
    cterminate
export usetup!, csetup!, udimen!, udimsh!, udimse!, uvartype!,
    unames!, ureport!, cdimen!, cdimsj!, cdimsh!, cdimse!, cstats!,
    cvartype!, cnames!, creport!, connames!, pname!, probname!, varnames!,
    ufn!, ugr!, uofg!, ubandh!, udh!, ush!, ueh!, ugrdh!, ugrsh!, ugreh!,
    uhprod!, cfn!, cofg!, cofsg!, ccfg!, clfg!, cgr!, csgr!, ccfsg!,
    ccifg!, ccifsg!, cgrdh!, cdh!, csh!, cshc!, ceh!, cidh!, cish!,
    csgrsh!, csgreh!, chprod!, chcprod!, cjprod!, uterminate!, cterminate!

"""
    x, x_l, x_u = usetup(input, out, io_buffer, n)

  - input:     [IN] Int
  - out:       [IN] Int
  - io_buffer: [IN] Int
  - n:         [IN] Int
  - x:         [OUT] Array{Float64, 1}
  - x_l:       [OUT] Array{Float64, 1}
  - x_u:       [OUT] Array{Float64, 1}
"""
function usetup(input::Int, out::Int, io_buffer::Int, n::Int)
  io_err = Cint[0]
  x = Array(Cdouble, n)
  x_l = Array(Cdouble, n)
  x_u = Array(Cdouble, n)
  usetup(io_err, Cint[input], Cint[out], Cint[io_buffer], Cint[n], x,
    x_l, x_u)
  @cutest_error
  return x, x_l, x_u
end

"""
    usetup!(input, out, io_buffer, n, x, x_l, x_u)

  - input:     [IN] Int
  - out:       [IN] Int
  - io_buffer: [IN] Int
  - n:         [IN] Int
  - x:         [OUT] Array{Float64, 1}
  - x_l:       [OUT] Array{Float64, 1}
  - x_u:       [OUT] Array{Float64, 1}
"""
function usetup!(input::Int, out::Int, io_buffer::Int, n::Int, x::Array{Float64, 1},
    x_l::Array{Float64, 1}, x_u::Array{Float64, 1})
  io_err = Cint[0]
  usetup(io_err, Cint[input], Cint[out], Cint[io_buffer], Cint[n], x,
    x_l, x_u)
  @cutest_error
  return
end

"""
    x, x_l, x_u, y, c_l, c_u, equatn, linear = csetup(input, out, io_buffer, n, m, e_order, l_order, v_order)

  - input:     [IN] Int
  - out:       [IN] Int
  - io_buffer: [IN] Int
  - n:         [IN] Int
  - m:         [IN] Int
  - x:         [OUT] Array{Float64, 1}
  - x_l:       [OUT] Array{Float64, 1}
  - x_u:       [OUT] Array{Float64, 1}
  - y:         [OUT] Array{Float64, 1}
  - c_l:       [OUT] Array{Float64, 1}
  - c_u:       [OUT] Array{Float64, 1}
  - equatn:    [OUT] Array{Bool, 1}
  - linear:    [OUT] Array{Bool, 1}
  - e_order:   [IN] Int
  - l_order:   [IN] Int
  - v_order:   [IN] Int
"""
function csetup(input::Int, out::Int, io_buffer::Int, n::Int, m::Int, e_order::Int,
    l_order::Int, v_order::Int)
  io_err = Cint[0]
  x = Array(Cdouble, n)
  x_l = Array(Cdouble, n)
  x_u = Array(Cdouble, n)
  y = Array(Cdouble, m)
  c_l = Array(Cdouble, m)
  c_u = Array(Cdouble, m)
  equatn = Array(Cint, m)
  linear = Array(Cint, m)
  csetup(io_err, Cint[input], Cint[out], Cint[io_buffer], Cint[n],
    Cint[m], x, x_l, x_u, y, c_l, c_u, equatn, linear, Cint[e_order],
    Cint[l_order], Cint[v_order])
  @cutest_error
  return x, x_l, x_u, y, c_l, c_u, equatn, linear
end

"""
    csetup!(input, out, io_buffer, n, m, x, x_l, x_u, y, c_l, c_u, equatn, linear, e_order, l_order, v_order)

  - input:     [IN] Int
  - out:       [IN] Int
  - io_buffer: [IN] Int
  - n:         [IN] Int
  - m:         [IN] Int
  - x:         [OUT] Array{Float64, 1}
  - x_l:       [OUT] Array{Float64, 1}
  - x_u:       [OUT] Array{Float64, 1}
  - y:         [OUT] Array{Float64, 1}
  - c_l:       [OUT] Array{Float64, 1}
  - c_u:       [OUT] Array{Float64, 1}
  - equatn:    [OUT] Array{Bool, 1}
  - linear:    [OUT] Array{Bool, 1}
  - e_order:   [IN] Int
  - l_order:   [IN] Int
  - v_order:   [IN] Int
"""
function csetup!(input::Int, out::Int, io_buffer::Int, n::Int, m::Int,
    x::Array{Float64, 1}, x_l::Array{Float64, 1}, x_u::Array{Float64, 1},
    y::Array{Float64, 1}, c_l::Array{Float64, 1}, c_u::Array{Float64, 1},
    equatn::Array{Bool, 1}, linear::Array{Bool, 1}, e_order::Int,
    l_order::Int, v_order::Int)
  io_err = Cint[0]
  csetup(io_err, Cint[input], Cint[out], Cint[io_buffer], Cint[n],
    Cint[m], x, x_l, x_u, y, c_l, c_u, equatn, linear, Cint[e_order],
    Cint[l_order], Cint[v_order])
  @cutest_error
  return
end

"""
    n = udimen(input)

  - input:   [IN] Int
  - n:       [OUT] Int
"""
function udimen(input::Int)
  io_err = Cint[0]
  n = Cint[0]
  udimen(io_err, Cint[input], n)
  @cutest_error
  return n[1]
end

"""
    nnzh = udimsh()

  - nnzh:    [OUT] Int
"""
function udimsh()
  io_err = Cint[0]
  nnzh = Cint[0]
  udimsh(io_err, nnzh)
  @cutest_error
  return nnzh[1]
end

"""
    ne, he_val_ne, he_row_ne = udimse()

  - ne:        [OUT] Int
  - he_val_ne: [OUT] Int
  - he_row_ne: [OUT] Int
"""
function udimse()
  io_err = Cint[0]
  ne = Cint[0]
  he_val_ne = Cint[0]
  he_row_ne = Cint[0]
  udimse(io_err, ne, he_val_ne, he_row_ne)
  @cutest_error
  return ne[1], he_val_ne[1], he_row_ne[1]
end

"""
    x_type = uvartype(n)

  - n:       [IN] Int
  - x_type:  [OUT] Array{Int, 1}
"""
function uvartype(n::Int)
  io_err = Cint[0]
  x_type = Array(Cint, n)
  uvartype(io_err, Cint[n], x_type)
  @cutest_error
  return x_type
end

"""
    uvartype!(n, x_type)

  - n:       [IN] Int
  - x_type:  [OUT] Array{Int, 1}
"""
function uvartype!(n::Int, x_type::Array{Int, 1})
  io_err = Cint[0]
  x_type_cp = Array(Cint, n)
  uvartype(io_err, Cint[n], x_type_cp)
  @cutest_error
  for i = 1:n
    x_type[i] = x_type_cp[i]
  end
  return
end

function uvartype!(n::Int, x_type::Array{Cint, 1})
  io_err = Cint[0]
  uvartype(io_err, Cint[n], x_type)
  @cutest_error
  return
end

"""
    pname, vname = unames(n)

  - n:       [IN] Int
  - pname:   [OUT] UInt8
  - vname:   [OUT] Array{UInt8, 1}
"""
function unames(n::Int)
  io_err = Cint[0]
  pname = Cchar[0]
  vname = Array(Cchar, n)
  unames(io_err, Cint[n], pname, vname)
  @cutest_error
  return pname[1], vname
end

"""
    pname = unames!(n, vname)

  - n:       [IN] Int
  - pname:   [OUT] UInt8
  - vname:   [OUT] Array{UInt8, 1}
"""
function unames!(n::Int, vname::Array{UInt8, 1})
  io_err = Cint[0]
  pname = Cchar[0]
  unames(io_err, Cint[n], pname, vname)
  @cutest_error
  return pname[1]
end

"""
    calls, time = ureport()

  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function ureport()
  io_err = Cint[0]
  calls = Array(Cdouble, 4)
  time = Array(Cdouble, 2)
  ureport(io_err, calls, time)
  @cutest_error
  return calls, time
end

"""
    ureport!(calls, time)

  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function ureport!(calls::Array{Float64, 1}, time::Array{Float64, 1})
  io_err = Cint[0]
  ureport(io_err, calls, time)
  @cutest_error
  return
end

"""
    calls, time = ureport(nlp)

  - nlp:     [IN] CUTEstModel
  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function ureport(nlp::CUTEstModel)
  io_err = Cint[0]
  calls = Array(Cdouble, 4)
  time = Array(Cdouble, 2)
  ureport(io_err, calls, time)
  @cutest_error
  return calls, time
end

"""
    ureport!(nlp, calls, time)

  - nlp:     [IN] CUTEstModel
  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function ureport!(nlp::CUTEstModel, calls::Array{Float64, 1}, time::Array{Float64, 1})
  io_err = Cint[0]
  ureport(io_err, calls, time)
  @cutest_error
  return
end

"""
    n, m = cdimen(input)

  - input:   [IN] Int
  - n:       [OUT] Int
  - m:       [OUT] Int
"""
function cdimen(input::Int)
  io_err = Cint[0]
  n = Cint[0]
  m = Cint[0]
  cdimen(io_err, Cint[input], n, m)
  @cutest_error
  return n[1], m[1]
end

"""
    nnzj = cdimsj()

  - nnzj:    [OUT] Int
"""
function cdimsj()
  io_err = Cint[0]
  nnzj = Cint[0]
  cdimsj(io_err, nnzj)
  @cutest_error
  return nnzj[1]
end

"""
    nnzh = cdimsh()

  - nnzh:    [OUT] Int
"""
function cdimsh()
  io_err = Cint[0]
  nnzh = Cint[0]
  cdimsh(io_err, nnzh)
  @cutest_error
  return nnzh[1]
end

"""
    ne, he_val_ne, he_row_ne = cdimse()

  - ne:        [OUT] Int
  - he_val_ne: [OUT] Int
  - he_row_ne: [OUT] Int
"""
function cdimse()
  io_err = Cint[0]
  ne = Cint[0]
  he_val_ne = Cint[0]
  he_row_ne = Cint[0]
  cdimse(io_err, ne, he_val_ne, he_row_ne)
  @cutest_error
  return ne[1], he_val_ne[1], he_row_ne[1]
end

"""
"""
function cstats()
  io_err = Cint[0]
  nonlinear_variables_objective = Cint[0]
  nonlinear_variables_constraints = Cint[0]
  equality_constraints = Cint[0]
  linear_constraints = Cint[0]
  cstats(io_err, nonlinear_variables_objective,
    nonlinear_variables_constraints, equality_constraints,
    linear_constraints)
  @cutest_error
  return nonlinear_variables_objective[1], nonlinear_variables_constraints[1], equality_constraints[1], linear_constraints[1]
end

"""
"""
function cstats(nlp::CUTEstModel)
  io_err = Cint[0]
  nonlinear_variables_objective = Cint[0]
  nonlinear_variables_constraints = Cint[0]
  equality_constraints = Cint[0]
  linear_constraints = Cint[0]
  cstats(io_err, nonlinear_variables_objective,
    nonlinear_variables_constraints, equality_constraints,
    linear_constraints)
  @cutest_error
  return nonlinear_variables_objective[1], nonlinear_variables_constraints[1], equality_constraints[1], linear_constraints[1]
end

"""
    x_type = cvartype(n)

  - n:       [IN] Int
  - x_type:  [OUT] Array{Int, 1}
"""
function cvartype(n::Int)
  io_err = Cint[0]
  x_type = Array(Cint, n)
  cvartype(io_err, Cint[n], x_type)
  @cutest_error
  return x_type
end

"""
    cvartype!(n, x_type)

  - n:       [IN] Int
  - x_type:  [OUT] Array{Int, 1}
"""
function cvartype!(n::Int, x_type::Array{Int, 1})
  io_err = Cint[0]
  x_type_cp = Array(Cint, n)
  cvartype(io_err, Cint[n], x_type_cp)
  @cutest_error
  for i = 1:n
    x_type[i] = x_type_cp[i]
  end
  return
end

function cvartype!(n::Int, x_type::Array{Cint, 1})
  io_err = Cint[0]
  cvartype(io_err, Cint[n], x_type)
  @cutest_error
  return
end

"""
    pname, vname, cname = cnames(n, m)

  - n:       [IN] Int
  - m:       [IN] Int
  - pname:   [OUT] UInt8
  - vname:   [OUT] Array{UInt8, 1}
  - cname:   [OUT] Array{UInt8, 1}
"""
function cnames(n::Int, m::Int)
  io_err = Cint[0]
  pname = Cchar[0]
  vname = Array(Cchar, n)
  cname = Array(Cchar, m)
  cnames(io_err, Cint[n], Cint[m], pname, vname, cname)
  @cutest_error
  return pname[1], vname, cname
end

"""
    pname = cnames!(n, m, vname, cname)

  - n:       [IN] Int
  - m:       [IN] Int
  - pname:   [OUT] UInt8
  - vname:   [OUT] Array{UInt8, 1}
  - cname:   [OUT] Array{UInt8, 1}
"""
function cnames!(n::Int, m::Int, vname::Array{UInt8, 1}, cname::Array{UInt8, 1})
  io_err = Cint[0]
  pname = Cchar[0]
  cnames(io_err, Cint[n], Cint[m], pname, vname, cname)
  @cutest_error
  return pname[1]
end

"""
    pname, vname, cname = cnames(nlp)

  - nlp:     [IN] CUTEstModel
  - pname:   [OUT] UInt8
  - vname:   [OUT] Array{UInt8, 1}
  - cname:   [OUT] Array{UInt8, 1}
"""
function cnames(nlp::CUTEstModel)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  pname = Cchar[0]
  vname = Array(Cchar, n)
  cname = Array(Cchar, m)
  cnames(io_err, Cint[n], Cint[m], pname, vname, cname)
  @cutest_error
  return pname[1], vname, cname
end

"""
    pname = cnames!(nlp, vname, cname)

  - nlp:     [IN] CUTEstModel
  - pname:   [OUT] UInt8
  - vname:   [OUT] Array{UInt8, 1}
  - cname:   [OUT] Array{UInt8, 1}
"""
function cnames!(nlp::CUTEstModel, vname::Array{UInt8, 1}, cname::Array{UInt8, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  pname = Cchar[0]
  cnames(io_err, Cint[n], Cint[m], pname, vname, cname)
  @cutest_error
  return pname[1]
end

"""
    calls, time = creport()

  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function creport()
  io_err = Cint[0]
  calls = Array(Cdouble, 7)
  time = Array(Cdouble, 2)
  creport(io_err, calls, time)
  @cutest_error
  return calls, time
end

"""
    creport!(calls, time)

  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function creport!(calls::Array{Float64, 1}, time::Array{Float64, 1})
  io_err = Cint[0]
  creport(io_err, calls, time)
  @cutest_error
  return
end

"""
    calls, time = creport(nlp)

  - nlp:     [IN] CUTEstModel
  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function creport(nlp::CUTEstModel)
  io_err = Cint[0]
  calls = Array(Cdouble, 7)
  time = Array(Cdouble, 2)
  creport(io_err, calls, time)
  @cutest_error
  return calls, time
end

"""
    creport!(nlp, calls, time)

  - nlp:     [IN] CUTEstModel
  - calls:   [OUT] Array{Float64, 1}
  - time:    [OUT] Array{Float64, 1}
"""
function creport!(nlp::CUTEstModel, calls::Array{Float64, 1}, time::Array{Float64, 1})
  io_err = Cint[0]
  creport(io_err, calls, time)
  @cutest_error
  return
end

"""
    cname = connames(m)

  - m:       [IN] Int
  - cname:   [OUT] Array{UInt8, 1}
"""
function connames(m::Int)
  io_err = Cint[0]
  cname = Array(Cchar, m)
  connames(io_err, Cint[m], cname)
  @cutest_error
  return cname
end

"""
    connames!(m, cname)

  - m:       [IN] Int
  - cname:   [OUT] Array{UInt8, 1}
"""
function connames!(m::Int, cname::Array{UInt8, 1})
  io_err = Cint[0]
  connames(io_err, Cint[m], cname)
  @cutest_error
  return
end

"""
    cname = connames(nlp)

  - nlp:     [IN] CUTEstModel
  - cname:   [OUT] Array{UInt8, 1}
"""
function connames(nlp::CUTEstModel)
  io_err = Cint[0]
  m = nlp.meta.ncon
  cname = Array(Cchar, m)
  connames(io_err, Cint[m], cname)
  @cutest_error
  return cname
end

"""
    connames!(nlp, cname)

  - nlp:     [IN] CUTEstModel
  - cname:   [OUT] Array{UInt8, 1}
"""
function connames!(nlp::CUTEstModel, cname::Array{UInt8, 1})
  io_err = Cint[0]
  m = nlp.meta.ncon
  connames(io_err, Cint[m], cname)
  @cutest_error
  return
end

"""
    pname = pname(input)

  - input:   [IN] Int
  - pname:   [OUT] UInt8
"""
function pname(input::Int)
  io_err = Cint[0]
  pname = Cchar[0]
  pname(io_err, Cint[input], pname)
  @cutest_error
  return pname[1]
end

"""
    pname = pname(nlp, input)

  - nlp:     [IN] CUTEstModel
  - input:   [IN] Int
  - pname:   [OUT] UInt8
"""
function pname(nlp::CUTEstModel, input::Int)
  io_err = Cint[0]
  pname = Cchar[0]
  pname(io_err, Cint[input], pname)
  @cutest_error
  return pname[1]
end

"""
    pname = probname()

  - pname:   [OUT] UInt8
"""
function probname()
  io_err = Cint[0]
  pname = Cchar[0]
  probname(io_err, pname)
  @cutest_error
  return pname[1]
end

"""
    pname = probname(nlp)

  - nlp:     [IN] CUTEstModel
  - pname:   [OUT] UInt8
"""
function probname(nlp::CUTEstModel)
  io_err = Cint[0]
  pname = Cchar[0]
  probname(io_err, pname)
  @cutest_error
  return pname[1]
end

"""
    vname = varnames(n)

  - n:       [IN] Int
  - vname:   [OUT] Array{UInt8, 1}
"""
function varnames(n::Int)
  io_err = Cint[0]
  vname = Array(Cchar, n)
  varnames(io_err, Cint[n], vname)
  @cutest_error
  return vname
end

"""
    varnames!(n, vname)

  - n:       [IN] Int
  - vname:   [OUT] Array{UInt8, 1}
"""
function varnames!(n::Int, vname::Array{UInt8, 1})
  io_err = Cint[0]
  varnames(io_err, Cint[n], vname)
  @cutest_error
  return
end

"""
    f = ufn(n, x)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
"""
function ufn(n::Int, x::Array{Float64, 1})
  io_err = Cint[0]
  f = Cdouble[0]
  ufn(io_err, Cint[n], x, f)
  @cutest_error
  return f[1]
end

"""
    f = ufn(nlp, x)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
"""
function ufn(nlp::CUTEstModel, x::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  ufn(io_err, Cint[n], x, f)
  @cutest_error
  return f[1]
end

"""
    g = ugr(n, x)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
"""
function ugr(n::Int, x::Array{Float64, 1})
  io_err = Cint[0]
  g = Array(Cdouble, n)
  ugr(io_err, Cint[n], x, g)
  @cutest_error
  return g
end

"""
    ugr!(n, x, g)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
"""
function ugr!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1})
  io_err = Cint[0]
  ugr(io_err, Cint[n], x, g)
  @cutest_error
  return
end

"""
    g = ugr(nlp, x)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
"""
function ugr(nlp::CUTEstModel, x::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  g = Array(Cdouble, n)
  ugr(io_err, Cint[n], x, g)
  @cutest_error
  return g
end

"""
    ugr!(nlp, x, g)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
"""
function ugr!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  ugr(io_err, Cint[n], x, g)
  @cutest_error
  return
end

"""
    f, g = uofg(n, x, grad)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function uofg(n::Int, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  g = Array(Cdouble, n)
  uofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1], g
end

"""
    f = uofg!(n, x, g, grad)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function uofg!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  uofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1]
end

"""
    f, g = uofg(nlp, x, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function uofg(nlp::CUTEstModel, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  g = Array(Cdouble, n)
  uofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1], g
end

"""
    f = uofg!(nlp, x, g, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function uofg!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  uofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1]
end

"""
    h_band, max_semibandwidth = ubandh(n, x, semibandwidth, lbandh)

  - n:                 [IN] Int
  - x:                 [IN] Array{Float64, 1}
  - semibandwidth:     [IN] Int
  - h_band:            [OUT] Array{Float64, 2}
  - lbandh:            [IN] Int
  - max_semibandwidth: [OUT] Int
"""
function ubandh(n::Int, x::Array{Float64, 1}, semibandwidth::Int, lbandh::Int)
  io_err = Cint[0]
  h_band = Array(Cdouble, lbandh - 0 + 1, n)
  max_semibandwidth = Cint[0]
  ubandh(io_err, Cint[n], x, Cint[semibandwidth], h_band,
    Cint[lbandh], max_semibandwidth)
  @cutest_error
  return h_band, max_semibandwidth[1]
end

"""
    max_semibandwidth = ubandh!(n, x, semibandwidth, h_band, lbandh)

  - n:                 [IN] Int
  - x:                 [IN] Array{Float64, 1}
  - semibandwidth:     [IN] Int
  - h_band:            [OUT] Array{Float64, 2}
  - lbandh:            [IN] Int
  - max_semibandwidth: [OUT] Int
"""
function ubandh!(n::Int, x::Array{Float64, 1}, semibandwidth::Int,
    h_band::Array{Float64, 2}, lbandh::Int)
  io_err = Cint[0]
  max_semibandwidth = Cint[0]
  ubandh(io_err, Cint[n], x, Cint[semibandwidth], h_band,
    Cint[lbandh], max_semibandwidth)
  @cutest_error
  return max_semibandwidth[1]
end

"""
    h_band, max_semibandwidth = ubandh(nlp, x, semibandwidth, lbandh)

  - nlp:               [IN] CUTEstModel
  - x:                 [IN] Array{Float64, 1}
  - semibandwidth:     [IN] Int
  - h_band:            [OUT] Array{Float64, 2}
  - lbandh:            [IN] Int
  - max_semibandwidth: [OUT] Int
"""
function ubandh(nlp::CUTEstModel, x::Array{Float64, 1}, semibandwidth::Int,
    lbandh::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  h_band = Array(Cdouble, lbandh - 0 + 1, n)
  max_semibandwidth = Cint[0]
  ubandh(io_err, Cint[n], x, Cint[semibandwidth], h_band,
    Cint[lbandh], max_semibandwidth)
  @cutest_error
  return h_band, max_semibandwidth[1]
end

"""
    max_semibandwidth = ubandh!(nlp, x, semibandwidth, h_band, lbandh)

  - nlp:               [IN] CUTEstModel
  - x:                 [IN] Array{Float64, 1}
  - semibandwidth:     [IN] Int
  - h_band:            [OUT] Array{Float64, 2}
  - lbandh:            [IN] Int
  - max_semibandwidth: [OUT] Int
"""
function ubandh!(nlp::CUTEstModel, x::Array{Float64, 1}, semibandwidth::Int,
    h_band::Array{Float64, 2}, lbandh::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  max_semibandwidth = Cint[0]
  ubandh(io_err, Cint[n], x, Cint[semibandwidth], h_band,
    Cint[lbandh], max_semibandwidth)
  @cutest_error
  return max_semibandwidth[1]
end

"""
    h = udh(n, x, lh1)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function udh(n::Int, x::Array{Float64, 1}, lh1::Int)
  io_err = Cint[0]
  h = Array(Cdouble, lh1, n)
  udh(io_err, Cint[n], x, Cint[lh1], h)
  @cutest_error
  return h
end

"""
    udh!(n, x, lh1, h)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function udh!(n::Int, x::Array{Float64, 1}, lh1::Int, h::Array{Float64, 2})
  io_err = Cint[0]
  udh(io_err, Cint[n], x, Cint[lh1], h)
  @cutest_error
  return
end

"""
    h = udh(nlp, x, lh1)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function udh(nlp::CUTEstModel, x::Array{Float64, 1}, lh1::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  h = Array(Cdouble, lh1, n)
  udh(io_err, Cint[n], x, Cint[lh1], h)
  @cutest_error
  return h
end

"""
    udh!(nlp, x, lh1, h)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function udh!(nlp::CUTEstModel, x::Array{Float64, 1}, lh1::Int, h::Array{Float64,
    2})
  io_err = Cint[0]
  n = nlp.meta.nvar
  udh(io_err, Cint[n], x, Cint[lh1], h)
  @cutest_error
  return
end

"""
    nnzh, h_val, h_row, h_col = ush(n, x, lh)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ush(n::Int, x::Array{Float64, 1}, lh::Int)
  io_err = Cint[0]
  nnzh = Cint[0]
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  ush(io_err, Cint[n], x, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = ush!(n, x, lh, h_val, h_row, h_col)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ush!(n::Int, x::Array{Float64, 1}, lh::Int, h_val::Array{Float64, 1},
    h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  nnzh = Cint[0]
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  ush(io_err, Cint[n], x, nnzh, Cint[lh], h_val, h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function ush!(n::Int, x::Array{Float64, 1}, lh::Int, h_val::Array{Float64, 1},
    h_row::Array{Cint, 1}, h_col::Array{Cint, 1})
  io_err = Cint[0]
  nnzh = Cint[0]
  ush(io_err, Cint[n], x, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzh[1]
end

"""
    nnzh, h_val, h_row, h_col = ush(nlp, x)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ush(nlp::CUTEstModel, x::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  ush(io_err, Cint[n], x, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = ush!(nlp, x, h_val, h_row, h_col)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ush!(nlp::CUTEstModel, x::Array{Float64, 1}, h_val::Array{Float64, 1},
    h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  ush(io_err, Cint[n], x, nnzh, Cint[lh], h_val, h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function ush!(nlp::CUTEstModel, x::Array{Float64, 1}, h_val::Array{Float64, 1},
    h_row::Array{Cint, 1}, h_col::Array{Cint, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  ush(io_err, Cint[n], x, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzh[1]
end

"""
    ne, he_row_ptr, he_val_ptr, he_row, he_val = ueh(n, x, lhe_ptr, lhe_row, lhe_val, byrows)

  - n:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ueh(n::Int, x::Array{Float64, 1}, lhe_ptr::Int, lhe_row::Int,
    lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  ueh(io_err, Cint[n], x, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    ne = ueh!(n, x, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - n:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ueh!(n::Int, x::Array{Float64, 1}, lhe_ptr::Int, he_row_ptr::Array{Int,
    1}, he_val_ptr::Array{Int, 1}, lhe_row::Int, he_row::Array{Int, 1},
    lhe_val::Int, he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  ueh(io_err, Cint[n], x, ne, Cint[lhe_ptr], he_row_ptr_cp,
    he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return ne[1]
end

function ueh!(n::Int, x::Array{Float64, 1}, lhe_ptr::Int, he_row_ptr::Array{Cint,
    1}, he_val_ptr::Array{Cint, 1}, lhe_row::Int, he_row::Array{Cint, 1},
    lhe_val::Int, he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  ueh(io_err, Cint[n], x, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return ne[1]
end

"""
    ne, he_row_ptr, he_val_ptr, he_row, he_val = ueh(nlp, x, lhe_ptr, lhe_row, lhe_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ueh(nlp::CUTEstModel, x::Array{Float64, 1}, lhe_ptr::Int, lhe_row::Int,
    lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  ueh(io_err, Cint[n], x, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    ne = ueh!(nlp, x, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ueh!(nlp::CUTEstModel, x::Array{Float64, 1}, lhe_ptr::Int,
    he_row_ptr::Array{Int, 1}, he_val_ptr::Array{Int, 1}, lhe_row::Int,
    he_row::Array{Int, 1}, lhe_val::Int, he_val::Array{Float64, 1},
    byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  ueh(io_err, Cint[n], x, ne, Cint[lhe_ptr], he_row_ptr_cp,
    he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return ne[1]
end

function ueh!(nlp::CUTEstModel, x::Array{Float64, 1}, lhe_ptr::Int,
    he_row_ptr::Array{Cint, 1}, he_val_ptr::Array{Cint, 1}, lhe_row::Int,
    he_row::Array{Cint, 1}, lhe_val::Int, he_val::Array{Float64, 1},
    byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ne = Cint[0]
  ueh(io_err, Cint[n], x, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return ne[1]
end

"""
    g, h = ugrdh(n, x, lh1)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function ugrdh(n::Int, x::Array{Float64, 1}, lh1::Int)
  io_err = Cint[0]
  g = Array(Cdouble, n)
  h = Array(Cdouble, lh1, n)
  ugrdh(io_err, Cint[n], x, g, Cint[lh1], h)
  @cutest_error
  return g, h
end

"""
    ugrdh!(n, x, g, lh1, h)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function ugrdh!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, lh1::Int,
    h::Array{Float64, 2})
  io_err = Cint[0]
  ugrdh(io_err, Cint[n], x, g, Cint[lh1], h)
  @cutest_error
  return
end

"""
    g, h = ugrdh(nlp, x, lh1)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function ugrdh(nlp::CUTEstModel, x::Array{Float64, 1}, lh1::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  g = Array(Cdouble, n)
  h = Array(Cdouble, lh1, n)
  ugrdh(io_err, Cint[n], x, g, Cint[lh1], h)
  @cutest_error
  return g, h
end

"""
    ugrdh!(nlp, x, g, lh1, h)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function ugrdh!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    lh1::Int, h::Array{Float64, 2})
  io_err = Cint[0]
  n = nlp.meta.nvar
  ugrdh(io_err, Cint[n], x, g, Cint[lh1], h)
  @cutest_error
  return
end

"""
    g, nnzh, h_val, h_row, h_col = ugrsh(n, x, lh)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ugrsh(n::Int, x::Array{Float64, 1}, lh::Int)
  io_err = Cint[0]
  g = Array(Cdouble, n)
  nnzh = Cint[0]
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  ugrsh(io_err, Cint[n], x, g, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return g, nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = ugrsh!(n, x, g, lh, h_val, h_row, h_col)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ugrsh!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  nnzh = Cint[0]
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  ugrsh(io_err, Cint[n], x, g, nnzh, Cint[lh], h_val, h_row_cp,
    h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function ugrsh!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  nnzh = Cint[0]
  ugrsh(io_err, Cint[n], x, g, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzh[1]
end

"""
    g, nnzh, h_val, h_row, h_col = ugrsh(nlp, x)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ugrsh(nlp::CUTEstModel, x::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  g = Array(Cdouble, n)
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  ugrsh(io_err, Cint[n], x, g, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return g, nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = ugrsh!(nlp, x, g, h_val, h_row, h_col)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - g:       [OUT] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function ugrsh!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  ugrsh(io_err, Cint[n], x, g, nnzh, Cint[lh], h_val, h_row_cp,
    h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function ugrsh!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  ugrsh(io_err, Cint[n], x, g, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzh[1]
end

"""
    g, ne, he_row_ptr, he_val_ptr, he_row, he_val = ugreh(n, x, lhe_ptr, lhe_row, lhe_val, byrows)

  - n:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - g:          [OUT] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ugreh(n::Int, x::Array{Float64, 1}, lhe_ptr::Int, lhe_row::Int,
    lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  g = Array(Cdouble, n)
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  ugreh(io_err, Cint[n], x, g, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return g, ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    ne = ugreh!(n, x, g, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - n:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - g:          [OUT] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ugreh!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, lhe_ptr::Int,
    he_row_ptr::Array{Int, 1}, he_val_ptr::Array{Int, 1}, lhe_row::Int,
    he_row::Array{Int, 1}, lhe_val::Int, he_val::Array{Float64, 1},
    byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  ugreh(io_err, Cint[n], x, g, ne, Cint[lhe_ptr], he_row_ptr_cp,
    he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return ne[1]
end

function ugreh!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, lhe_ptr::Int,
    he_row_ptr::Array{Cint, 1}, he_val_ptr::Array{Cint, 1}, lhe_row::Int,
    he_row::Array{Cint, 1}, lhe_val::Int, he_val::Array{Float64, 1},
    byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  ugreh(io_err, Cint[n], x, g, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return ne[1]
end

"""
    g, ne, he_row_ptr, he_val_ptr, he_row, he_val = ugreh(nlp, x, lhe_ptr, lhe_row, lhe_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - g:          [OUT] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ugreh(nlp::CUTEstModel, x::Array{Float64, 1}, lhe_ptr::Int, lhe_row::Int,
    lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  g = Array(Cdouble, n)
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  ugreh(io_err, Cint[n], x, g, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return g, ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    ne = ugreh!(nlp, x, g, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - g:          [OUT] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ugreh!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    lhe_ptr::Int, he_row_ptr::Array{Int, 1}, he_val_ptr::Array{Int, 1},
    lhe_row::Int, he_row::Array{Int, 1}, lhe_val::Int,
    he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  ugreh(io_err, Cint[n], x, g, ne, Cint[lhe_ptr], he_row_ptr_cp,
    he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return ne[1]
end

function ugreh!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    lhe_ptr::Int, he_row_ptr::Array{Cint, 1}, he_val_ptr::Array{Cint, 1},
    lhe_row::Int, he_row::Array{Cint, 1}, lhe_val::Int,
    he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ne = Cint[0]
  ugreh(io_err, Cint[n], x, g, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return ne[1]
end

"""
    result = uhprod(n, goth, x, vector)

  - n:       [IN] Int
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function uhprod(n::Int, goth::Bool, x::Array{Float64, 1}, vector::Array{Float64, 1})
  io_err = Cint[0]
  result = Array(Cdouble, n)
  uhprod(io_err, Cint[n], Cint[goth], x, vector, result)
  @cutest_error
  return result
end

"""
    uhprod!(n, goth, x, vector, result)

  - n:       [IN] Int
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function uhprod!(n::Int, goth::Bool, x::Array{Float64, 1}, vector::Array{Float64, 1},
    result::Array{Float64, 1})
  io_err = Cint[0]
  uhprod(io_err, Cint[n], Cint[goth], x, vector, result)
  @cutest_error
  return
end

"""
    result = uhprod(nlp, goth, x, vector)

  - nlp:     [IN] CUTEstModel
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function uhprod(nlp::CUTEstModel, goth::Bool, x::Array{Float64, 1},
    vector::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  result = Array(Cdouble, n)
  uhprod(io_err, Cint[n], Cint[goth], x, vector, result)
  @cutest_error
  return result
end

"""
    uhprod!(nlp, goth, x, vector, result)

  - nlp:     [IN] CUTEstModel
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function uhprod!(nlp::CUTEstModel, goth::Bool, x::Array{Float64, 1},
    vector::Array{Float64, 1}, result::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  uhprod(io_err, Cint[n], Cint[goth], x, vector, result)
  @cutest_error
  return
end

"""
    f, c = cfn(n, m, x)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - c:       [OUT] Array{Float64, 1}
"""
function cfn(n::Int, m::Int, x::Array{Float64, 1})
  io_err = Cint[0]
  f = Cdouble[0]
  c = Array(Cdouble, m)
  cfn(io_err, Cint[n], Cint[m], x, f, c)
  @cutest_error
  return f[1], c
end

"""
    f = cfn!(n, m, x, c)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - c:       [OUT] Array{Float64, 1}
"""
function cfn!(n::Int, m::Int, x::Array{Float64, 1}, c::Array{Float64, 1})
  io_err = Cint[0]
  f = Cdouble[0]
  cfn(io_err, Cint[n], Cint[m], x, f, c)
  @cutest_error
  return f[1]
end

"""
    f, c = cfn(nlp, x)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - c:       [OUT] Array{Float64, 1}
"""
function cfn(nlp::CUTEstModel, x::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  f = Cdouble[0]
  c = Array(Cdouble, m)
  cfn(io_err, Cint[n], Cint[m], x, f, c)
  @cutest_error
  return f[1], c
end

"""
    f = cfn!(nlp, x, c)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - c:       [OUT] Array{Float64, 1}
"""
function cfn!(nlp::CUTEstModel, x::Array{Float64, 1}, c::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  f = Cdouble[0]
  cfn(io_err, Cint[n], Cint[m], x, f, c)
  @cutest_error
  return f[1]
end

"""
    f, g = cofg(n, x, grad)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function cofg(n::Int, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  g = Array(Cdouble, n)
  cofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1], g
end

"""
    f = cofg!(n, x, g, grad)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function cofg!(n::Int, x::Array{Float64, 1}, g::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  cofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1]
end

"""
    f, g = cofg(nlp, x, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function cofg(nlp::CUTEstModel, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  g = Array(Cdouble, n)
  cofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1], g
end

"""
    f = cofg!(nlp, x, g, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function cofg!(nlp::CUTEstModel, x::Array{Float64, 1}, g::Array{Float64, 1},
    grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  cofg(io_err, Cint[n], x, f, g, Cint[grad])
  @cutest_error
  return f[1]
end

"""
    f, nnzg, g_val, g_var = cofsg(n, x, lg, grad)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - nnzg:    [OUT] Int
  - lg:      [IN] Int
  - g_val:   [OUT] Array{Float64, 1}
  - g_var:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function cofsg(n::Int, x::Array{Float64, 1}, lg::Int, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  nnzg = Cint[0]
  g_val = Array(Cdouble, lg)
  g_var = Array(Cint, lg)
  cofsg(io_err, Cint[n], x, f, nnzg, Cint[lg], g_val, g_var,
    Cint[grad])
  @cutest_error
  return f[1], nnzg[1], g_val, g_var
end

"""
    f, nnzg = cofsg!(n, x, lg, g_val, g_var, grad)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - nnzg:    [OUT] Int
  - lg:      [IN] Int
  - g_val:   [OUT] Array{Float64, 1}
  - g_var:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function cofsg!(n::Int, x::Array{Float64, 1}, lg::Int, g_val::Array{Float64, 1},
    g_var::Array{Int, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  nnzg = Cint[0]
  g_var_cp = Array(Cint, lg)
  cofsg(io_err, Cint[n], x, f, nnzg, Cint[lg], g_val, g_var_cp,
    Cint[grad])
  @cutest_error
  for i = 1:lg
    g_var[i] = g_var_cp[i]
  end
  return f[1], nnzg[1]
end

function cofsg!(n::Int, x::Array{Float64, 1}, lg::Int, g_val::Array{Float64, 1},
    g_var::Array{Cint, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  nnzg = Cint[0]
  cofsg(io_err, Cint[n], x, f, nnzg, Cint[lg], g_val, g_var,
    Cint[grad])
  @cutest_error
  return f[1], nnzg[1]
end

"""
    f, nnzg, g_val, g_var = cofsg(nlp, x, lg, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - nnzg:    [OUT] Int
  - lg:      [IN] Int
  - g_val:   [OUT] Array{Float64, 1}
  - g_var:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function cofsg(nlp::CUTEstModel, x::Array{Float64, 1}, lg::Int, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  nnzg = Cint[0]
  g_val = Array(Cdouble, lg)
  g_var = Array(Cint, lg)
  cofsg(io_err, Cint[n], x, f, nnzg, Cint[lg], g_val, g_var,
    Cint[grad])
  @cutest_error
  return f[1], nnzg[1], g_val, g_var
end

"""
    f, nnzg = cofsg!(nlp, x, lg, g_val, g_var, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - nnzg:    [OUT] Int
  - lg:      [IN] Int
  - g_val:   [OUT] Array{Float64, 1}
  - g_var:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function cofsg!(nlp::CUTEstModel, x::Array{Float64, 1}, lg::Int,
    g_val::Array{Float64, 1}, g_var::Array{Int, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  nnzg = Cint[0]
  g_var_cp = Array(Cint, lg)
  cofsg(io_err, Cint[n], x, f, nnzg, Cint[lg], g_val, g_var_cp,
    Cint[grad])
  @cutest_error
  for i = 1:lg
    g_var[i] = g_var_cp[i]
  end
  return f[1], nnzg[1]
end

function cofsg!(nlp::CUTEstModel, x::Array{Float64, 1}, lg::Int,
    g_val::Array{Float64, 1}, g_var::Array{Cint, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  f = Cdouble[0]
  nnzg = Cint[0]
  cofsg(io_err, Cint[n], x, f, nnzg, Cint[lg], g_val, g_var,
    Cint[grad])
  @cutest_error
  return f[1], nnzg[1]
end

"""
    c, cjac = ccfg(n, m, x, jtrans, lcjac1, lcjac2, grad)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lcjac1:  [IN] Int
  - lcjac2:  [IN] Int
  - cjac:    [OUT] Array{Float64, 2}
  - grad:    [IN] Bool
"""
function ccfg(n::Int, m::Int, x::Array{Float64, 1}, jtrans::Bool, lcjac1::Int,
    lcjac2::Int, grad::Bool)
  io_err = Cint[0]
  c = Array(Cdouble, m)
  cjac = Array(Cdouble, lcjac1, lcjac2)
  ccfg(io_err, Cint[n], Cint[m], x, c, Cint[jtrans], Cint[lcjac1],
    Cint[lcjac2], cjac, Cint[grad])
  @cutest_error
  return c, cjac
end

"""
    ccfg!(n, m, x, c, jtrans, lcjac1, lcjac2, cjac, grad)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lcjac1:  [IN] Int
  - lcjac2:  [IN] Int
  - cjac:    [OUT] Array{Float64, 2}
  - grad:    [IN] Bool
"""
function ccfg!(n::Int, m::Int, x::Array{Float64, 1}, c::Array{Float64, 1},
    jtrans::Bool, lcjac1::Int, lcjac2::Int, cjac::Array{Float64, 2},
    grad::Bool)
  io_err = Cint[0]
  ccfg(io_err, Cint[n], Cint[m], x, c, Cint[jtrans], Cint[lcjac1],
    Cint[lcjac2], cjac, Cint[grad])
  @cutest_error
  return
end

"""
    c, cjac = ccfg(nlp, x, jtrans, lcjac1, lcjac2, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lcjac1:  [IN] Int
  - lcjac2:  [IN] Int
  - cjac:    [OUT] Array{Float64, 2}
  - grad:    [IN] Bool
"""
function ccfg(nlp::CUTEstModel, x::Array{Float64, 1}, jtrans::Bool, lcjac1::Int,
    lcjac2::Int, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  c = Array(Cdouble, m)
  cjac = Array(Cdouble, lcjac1, lcjac2)
  ccfg(io_err, Cint[n], Cint[m], x, c, Cint[jtrans], Cint[lcjac1],
    Cint[lcjac2], cjac, Cint[grad])
  @cutest_error
  return c, cjac
end

"""
    ccfg!(nlp, x, c, jtrans, lcjac1, lcjac2, cjac, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lcjac1:  [IN] Int
  - lcjac2:  [IN] Int
  - cjac:    [OUT] Array{Float64, 2}
  - grad:    [IN] Bool
"""
function ccfg!(nlp::CUTEstModel, x::Array{Float64, 1}, c::Array{Float64, 1},
    jtrans::Bool, lcjac1::Int, lcjac2::Int, cjac::Array{Float64, 2},
    grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  ccfg(io_err, Cint[n], Cint[m], x, c, Cint[jtrans], Cint[lcjac1],
    Cint[lcjac2], cjac, Cint[grad])
  @cutest_error
  return
end

"""
    f, g = clfg(n, m, x, y, grad)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function clfg(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  g = Array(Cdouble, n)
  clfg(io_err, Cint[n], Cint[m], x, y, f, g, Cint[grad])
  @cutest_error
  return f[1], g
end

"""
    f = clfg!(n, m, x, y, g, grad)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function clfg!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    g::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  f = Cdouble[0]
  clfg(io_err, Cint[n], Cint[m], x, y, f, g, Cint[grad])
  @cutest_error
  return f[1]
end

"""
    f, g = clfg(nlp, x, y, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function clfg(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  f = Cdouble[0]
  g = Array(Cdouble, n)
  clfg(io_err, Cint[n], Cint[m], x, y, f, g, Cint[grad])
  @cutest_error
  return f[1], g
end

"""
    f = clfg!(nlp, x, y, g, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - f:       [OUT] Float64
  - g:       [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function clfg!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    g::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  f = Cdouble[0]
  clfg(io_err, Cint[n], Cint[m], x, y, f, g, Cint[grad])
  @cutest_error
  return f[1]
end

"""
    g, j_val = cgr(n, m, x, y, grlagf, jtrans, lj1, lj2)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
"""
function cgr(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, jtrans::Bool, lj1::Int, lj2::Int)
  io_err = Cint[0]
  g = Array(Cdouble, n)
  j_val = Array(Cdouble, lj1, lj2)
  cgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val)
  @cutest_error
  return g, j_val
end

"""
    cgr!(n, m, x, y, grlagf, g, jtrans, lj1, lj2, j_val)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
"""
function cgr!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, g::Array{Float64, 1}, jtrans::Bool, lj1::Int, lj2::Int,
    j_val::Array{Float64, 2})
  io_err = Cint[0]
  cgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val)
  @cutest_error
  return
end

"""
    g, j_val = cgr(nlp, x, y, grlagf, jtrans, lj1, lj2)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
"""
function cgr(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, jtrans::Bool, lj1::Int, lj2::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  g = Array(Cdouble, n)
  j_val = Array(Cdouble, lj1, lj2)
  cgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val)
  @cutest_error
  return g, j_val
end

"""
    cgr!(nlp, x, y, grlagf, g, jtrans, lj1, lj2, j_val)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
"""
function cgr!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, g::Array{Float64, 1}, jtrans::Bool, lj1::Int, lj2::Int,
    j_val::Array{Float64, 2})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  cgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val)
  @cutest_error
  return
end

"""
    nnzj, j_val, j_var, j_fun = csgr(n, m, x, y, grlagf, lj)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - lj:      [IN] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
"""
function csgr(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int)
  io_err = Cint[0]
  nnzj = Cint[0]
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  csgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun)
  @cutest_error
  return nnzj[1], j_val, j_var, j_fun
end

"""
    nnzj = csgr!(n, m, x, y, grlagf, lj, j_val, j_var, j_fun)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - lj:      [IN] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
"""
function csgr!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, j_val::Array{Float64, 1}, j_var::Array{Int, 1},
    j_fun::Array{Int, 1})
  io_err = Cint[0]
  nnzj = Cint[0]
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  csgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var_cp, j_fun_cp)
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  return nnzj[1]
end

function csgr!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, j_val::Array{Float64, 1}, j_var::Array{Cint,
    1}, j_fun::Array{Cint, 1})
  io_err = Cint[0]
  nnzj = Cint[0]
  csgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun)
  @cutest_error
  return nnzj[1]
end

"""
    nnzj, j_val, j_var, j_fun = csgr(nlp, x, y, grlagf)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
"""
function csgr(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj + nlp.meta.nvar
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  csgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun)
  @cutest_error
  return nnzj[1], j_val, j_var, j_fun
end

"""
    nnzj = csgr!(nlp, x, y, grlagf, j_val, j_var, j_fun)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
"""
function csgr!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, j_val::Array{Float64, 1}, j_var::Array{Int, 1},
    j_fun::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj + nlp.meta.nvar
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  csgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var_cp, j_fun_cp)
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  return nnzj[1]
end

function csgr!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, j_val::Array{Float64, 1}, j_var::Array{Cint, 1},
    j_fun::Array{Cint, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj + nlp.meta.nvar
  csgr(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun)
  @cutest_error
  return nnzj[1]
end

"""
    c, nnzj, j_val, j_var, j_fun = ccfsg(n, m, x, lj, grad)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - nnzj:    [OUT] Int
  - lj:      [IN] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccfsg(n::Int, m::Int, x::Array{Float64, 1}, lj::Int, grad::Bool)
  io_err = Cint[0]
  c = Array(Cdouble, m)
  nnzj = Cint[0]
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  ccfsg(io_err, Cint[n], Cint[m], x, c, nnzj, Cint[lj], j_val, j_var,
    j_fun, Cint[grad])
  @cutest_error
  return c, nnzj[1], j_val, j_var, j_fun
end

"""
    nnzj = ccfsg!(n, m, x, c, lj, j_val, j_var, j_fun, grad)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - nnzj:    [OUT] Int
  - lj:      [IN] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccfsg!(n::Int, m::Int, x::Array{Float64, 1}, c::Array{Float64, 1}, lj::Int,
    j_val::Array{Float64, 1}, j_var::Array{Int, 1}, j_fun::Array{Int, 1},
    grad::Bool)
  io_err = Cint[0]
  nnzj = Cint[0]
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  ccfsg(io_err, Cint[n], Cint[m], x, c, nnzj, Cint[lj], j_val,
    j_var_cp, j_fun_cp, Cint[grad])
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  return nnzj[1]
end

function ccfsg!(n::Int, m::Int, x::Array{Float64, 1}, c::Array{Float64, 1}, lj::Int,
    j_val::Array{Float64, 1}, j_var::Array{Cint, 1}, j_fun::Array{Cint,
    1}, grad::Bool)
  io_err = Cint[0]
  nnzj = Cint[0]
  ccfsg(io_err, Cint[n], Cint[m], x, c, nnzj, Cint[lj], j_val, j_var,
    j_fun, Cint[grad])
  @cutest_error
  return nnzj[1]
end

"""
    c, nnzj, j_val, j_var, j_fun = ccfsg(nlp, x, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - nnzj:    [OUT] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccfsg(nlp::CUTEstModel, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  c = Array(Cdouble, m)
  nnzj = Cint[0]
  lj = nlp.meta.nnzj
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  ccfsg(io_err, Cint[n], Cint[m], x, c, nnzj, Cint[lj], j_val, j_var,
    j_fun, Cint[grad])
  @cutest_error
  return c, nnzj[1], j_val, j_var, j_fun
end

"""
    nnzj = ccfsg!(nlp, x, c, j_val, j_var, j_fun, grad)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - c:       [OUT] Array{Float64, 1}
  - nnzj:    [OUT] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccfsg!(nlp::CUTEstModel, x::Array{Float64, 1}, c::Array{Float64, 1},
    j_val::Array{Float64, 1}, j_var::Array{Int, 1}, j_fun::Array{Int, 1},
    grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  ccfsg(io_err, Cint[n], Cint[m], x, c, nnzj, Cint[lj], j_val,
    j_var_cp, j_fun_cp, Cint[grad])
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  return nnzj[1]
end

function ccfsg!(nlp::CUTEstModel, x::Array{Float64, 1}, c::Array{Float64, 1},
    j_val::Array{Float64, 1}, j_var::Array{Cint, 1}, j_fun::Array{Cint,
    1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj
  ccfsg(io_err, Cint[n], Cint[m], x, c, nnzj, Cint[lj], j_val, j_var,
    j_fun, Cint[grad])
  @cutest_error
  return nnzj[1]
end

"""
    ci, gci = ccifg(n, icon, x, grad)

  - n:       [IN] Int
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - gci:     [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function ccifg(n::Int, icon::Int, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  ci = Cdouble[0]
  gci = Array(Cdouble, n)
  ccifg(io_err, Cint[n], Cint[icon], x, ci, gci, Cint[grad])
  @cutest_error
  return ci[1], gci
end

"""
    ci = ccifg!(n, icon, x, gci, grad)

  - n:       [IN] Int
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - gci:     [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function ccifg!(n::Int, icon::Int, x::Array{Float64, 1}, gci::Array{Float64, 1},
    grad::Bool)
  io_err = Cint[0]
  ci = Cdouble[0]
  ccifg(io_err, Cint[n], Cint[icon], x, ci, gci, Cint[grad])
  @cutest_error
  return ci[1]
end

"""
    ci, gci = ccifg(nlp, icon, x, grad)

  - nlp:     [IN] CUTEstModel
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - gci:     [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function ccifg(nlp::CUTEstModel, icon::Int, x::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ci = Cdouble[0]
  gci = Array(Cdouble, n)
  ccifg(io_err, Cint[n], Cint[icon], x, ci, gci, Cint[grad])
  @cutest_error
  return ci[1], gci
end

"""
    ci = ccifg!(nlp, icon, x, gci, grad)

  - nlp:     [IN] CUTEstModel
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - gci:     [OUT] Array{Float64, 1}
  - grad:    [IN] Bool
"""
function ccifg!(nlp::CUTEstModel, icon::Int, x::Array{Float64, 1},
    gci::Array{Float64, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ci = Cdouble[0]
  ccifg(io_err, Cint[n], Cint[icon], x, ci, gci, Cint[grad])
  @cutest_error
  return ci[1]
end

"""
    ci, nnzgci, gci_val, gci_var = ccifsg(n, icon, x, lgci, grad)

  - n:       [IN] Int
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - nnzgci:  [OUT] Int
  - lgci:    [IN] Int
  - gci_val: [OUT] Array{Float64, 1}
  - gci_var: [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccifsg(n::Int, icon::Int, x::Array{Float64, 1}, lgci::Int, grad::Bool)
  io_err = Cint[0]
  ci = Cdouble[0]
  nnzgci = Cint[0]
  gci_val = Array(Cdouble, lgci)
  gci_var = Array(Cint, lgci)
  ccifsg(io_err, Cint[n], Cint[icon], x, ci, nnzgci, Cint[lgci],
    gci_val, gci_var, Cint[grad])
  @cutest_error
  return ci[1], nnzgci[1], gci_val, gci_var
end

"""
    ci, nnzgci = ccifsg!(n, icon, x, lgci, gci_val, gci_var, grad)

  - n:       [IN] Int
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - nnzgci:  [OUT] Int
  - lgci:    [IN] Int
  - gci_val: [OUT] Array{Float64, 1}
  - gci_var: [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccifsg!(n::Int, icon::Int, x::Array{Float64, 1}, lgci::Int,
    gci_val::Array{Float64, 1}, gci_var::Array{Int, 1}, grad::Bool)
  io_err = Cint[0]
  ci = Cdouble[0]
  nnzgci = Cint[0]
  gci_var_cp = Array(Cint, lgci)
  ccifsg(io_err, Cint[n], Cint[icon], x, ci, nnzgci, Cint[lgci],
    gci_val, gci_var_cp, Cint[grad])
  @cutest_error
  for i = 1:lgci
    gci_var[i] = gci_var_cp[i]
  end
  return ci[1], nnzgci[1]
end

function ccifsg!(n::Int, icon::Int, x::Array{Float64, 1}, lgci::Int,
    gci_val::Array{Float64, 1}, gci_var::Array{Cint, 1}, grad::Bool)
  io_err = Cint[0]
  ci = Cdouble[0]
  nnzgci = Cint[0]
  ccifsg(io_err, Cint[n], Cint[icon], x, ci, nnzgci, Cint[lgci],
    gci_val, gci_var, Cint[grad])
  @cutest_error
  return ci[1], nnzgci[1]
end

"""
    ci, nnzgci, gci_val, gci_var = ccifsg(nlp, icon, x, lgci, grad)

  - nlp:     [IN] CUTEstModel
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - nnzgci:  [OUT] Int
  - lgci:    [IN] Int
  - gci_val: [OUT] Array{Float64, 1}
  - gci_var: [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccifsg(nlp::CUTEstModel, icon::Int, x::Array{Float64, 1}, lgci::Int,
    grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ci = Cdouble[0]
  nnzgci = Cint[0]
  gci_val = Array(Cdouble, lgci)
  gci_var = Array(Cint, lgci)
  ccifsg(io_err, Cint[n], Cint[icon], x, ci, nnzgci, Cint[lgci],
    gci_val, gci_var, Cint[grad])
  @cutest_error
  return ci[1], nnzgci[1], gci_val, gci_var
end

"""
    ci, nnzgci = ccifsg!(nlp, icon, x, lgci, gci_val, gci_var, grad)

  - nlp:     [IN] CUTEstModel
  - icon:    [IN] Int
  - x:       [IN] Array{Float64, 1}
  - ci:      [OUT] Float64
  - nnzgci:  [OUT] Int
  - lgci:    [IN] Int
  - gci_val: [OUT] Array{Float64, 1}
  - gci_var: [OUT] Array{Int, 1}
  - grad:    [IN] Bool
"""
function ccifsg!(nlp::CUTEstModel, icon::Int, x::Array{Float64, 1}, lgci::Int,
    gci_val::Array{Float64, 1}, gci_var::Array{Int, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ci = Cdouble[0]
  nnzgci = Cint[0]
  gci_var_cp = Array(Cint, lgci)
  ccifsg(io_err, Cint[n], Cint[icon], x, ci, nnzgci, Cint[lgci],
    gci_val, gci_var_cp, Cint[grad])
  @cutest_error
  for i = 1:lgci
    gci_var[i] = gci_var_cp[i]
  end
  return ci[1], nnzgci[1]
end

function ccifsg!(nlp::CUTEstModel, icon::Int, x::Array{Float64, 1}, lgci::Int,
    gci_val::Array{Float64, 1}, gci_var::Array{Cint, 1}, grad::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  ci = Cdouble[0]
  nnzgci = Cint[0]
  ccifsg(io_err, Cint[n], Cint[icon], x, ci, nnzgci, Cint[lgci],
    gci_val, gci_var, Cint[grad])
  @cutest_error
  return ci[1], nnzgci[1]
end

"""
    g, j_val, h_val = cgrdh(n, m, x, y, grlagf, jtrans, lj1, lj2, lh1)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cgrdh(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, jtrans::Bool, lj1::Int, lj2::Int, lh1::Int)
  io_err = Cint[0]
  g = Array(Cdouble, n)
  j_val = Array(Cdouble, lj1, lj2)
  h_val = Array(Cdouble, lh1, n)
  cgrdh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val, Cint[lh1], h_val)
  @cutest_error
  return g, j_val, h_val
end

"""
    cgrdh!(n, m, x, y, grlagf, g, jtrans, lj1, lj2, j_val, lh1, h_val)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cgrdh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, g::Array{Float64, 1}, jtrans::Bool, lj1::Int, lj2::Int,
    j_val::Array{Float64, 2}, lh1::Int, h_val::Array{Float64, 2})
  io_err = Cint[0]
  cgrdh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val, Cint[lh1], h_val)
  @cutest_error
  return
end

"""
    g, j_val, h_val = cgrdh(nlp, x, y, grlagf, jtrans, lj1, lj2, lh1)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cgrdh(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, jtrans::Bool, lj1::Int, lj2::Int, lh1::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  g = Array(Cdouble, n)
  j_val = Array(Cdouble, lj1, lj2)
  h_val = Array(Cdouble, lh1, n)
  cgrdh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val, Cint[lh1], h_val)
  @cutest_error
  return g, j_val, h_val
end

"""
    cgrdh!(nlp, x, y, grlagf, g, jtrans, lj1, lj2, j_val, lh1, h_val)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - g:       [OUT] Array{Float64, 1}
  - jtrans:  [IN] Bool
  - lj1:     [IN] Int
  - lj2:     [IN] Int
  - j_val:   [OUT] Array{Float64, 2}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cgrdh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, g::Array{Float64, 1}, jtrans::Bool, lj1::Int, lj2::Int,
    j_val::Array{Float64, 2}, lh1::Int, h_val::Array{Float64, 2})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  cgrdh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], g, Cint[jtrans],
    Cint[lj1], Cint[lj2], j_val, Cint[lh1], h_val)
  @cutest_error
  return
end

"""
    h_val = cdh(n, m, x, y, lh1)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cdh(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh1::Int)
  io_err = Cint[0]
  h_val = Array(Cdouble, lh1, n)
  cdh(io_err, Cint[n], Cint[m], x, y, Cint[lh1], h_val)
  @cutest_error
  return h_val
end

"""
    cdh!(n, m, x, y, lh1, h_val)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cdh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh1::Int,
    h_val::Array{Float64, 2})
  io_err = Cint[0]
  cdh(io_err, Cint[n], Cint[m], x, y, Cint[lh1], h_val)
  @cutest_error
  return
end

"""
    h_val = cdh(nlp, x, y, lh1)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cdh(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    lh1::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  h_val = Array(Cdouble, lh1, n)
  cdh(io_err, Cint[n], Cint[m], x, y, Cint[lh1], h_val)
  @cutest_error
  return h_val
end

"""
    cdh!(nlp, x, y, lh1, h_val)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - lh1:     [IN] Int
  - h_val:   [OUT] Array{Float64, 2}
"""
function cdh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    lh1::Int, h_val::Array{Float64, 2})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  cdh(io_err, Cint[n], Cint[m], x, y, Cint[lh1], h_val)
  @cutest_error
  return
end

"""
    nnzh, h_val, h_row, h_col = csh(n, m, x, y, lh)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csh(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh::Int)
  io_err = Cint[0]
  nnzh = Cint[0]
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  csh(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = csh!(n, m, x, y, lh, h_val, h_row, h_col)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  nnzh = Cint[0]
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  csh(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row_cp,
    h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function csh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  nnzh = Cint[0]
  csh(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1]
end

"""
    nnzh, h_val, h_row, h_col = csh(nlp, x, y)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csh(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  csh(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = csh!(nlp, x, y, h_val, h_row, h_col)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  csh(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row_cp,
    h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function csh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  csh(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1]
end

"""
    nnzh, h_val, h_row, h_col = cshc(n, m, x, y, lh)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cshc(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh::Int)
  io_err = Cint[0]
  nnzh = Cint[0]
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  cshc(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = cshc!(n, m, x, y, lh, h_val, h_row, h_col)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cshc!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  nnzh = Cint[0]
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  cshc(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val,
    h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function cshc!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1}, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  nnzh = Cint[0]
  cshc(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1]
end

"""
    nnzh, h_val, h_row, h_col = cshc(nlp, x, y)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cshc(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  cshc(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = cshc!(nlp, x, y, h_val, h_row, h_col)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cshc!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  cshc(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val,
    h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function cshc!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  cshc(io_err, Cint[n], Cint[m], x, y, nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1]
end

"""
    ne, he_row_ptr, he_val_ptr, he_row, he_val = ceh(n, m, x, y, lhe_ptr, lhe_row, lhe_val, byrows)

  - n:          [IN] Int
  - m:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ceh(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    lhe_ptr::Int, lhe_row::Int, lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  ceh(io_err, Cint[n], Cint[m], x, y, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    ne = ceh!(n, m, x, y, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - n:          [IN] Int
  - m:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ceh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    lhe_ptr::Int, he_row_ptr::Array{Int, 1}, he_val_ptr::Array{Int, 1},
    lhe_row::Int, he_row::Array{Int, 1}, lhe_val::Int,
    he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  ceh(io_err, Cint[n], Cint[m], x, y, ne, Cint[lhe_ptr],
    he_row_ptr_cp, he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val],
    he_val, Cint[byrows])
  @cutest_error
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return ne[1]
end

function ceh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    lhe_ptr::Int, he_row_ptr::Array{Cint, 1}, he_val_ptr::Array{Cint, 1},
    lhe_row::Int, he_row::Array{Cint, 1}, lhe_val::Int,
    he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  ne = Cint[0]
  ceh(io_err, Cint[n], Cint[m], x, y, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return ne[1]
end

"""
    ne, he_row_ptr, he_val_ptr, he_row, he_val = ceh(nlp, x, y, lhe_ptr, lhe_row, lhe_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ceh(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    lhe_ptr::Int, lhe_row::Int, lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  ceh(io_err, Cint[n], Cint[m], x, y, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    ne = ceh!(nlp, x, y, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function ceh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    lhe_ptr::Int, he_row_ptr::Array{Int, 1}, he_val_ptr::Array{Int, 1},
    lhe_row::Int, he_row::Array{Int, 1}, lhe_val::Int,
    he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  ceh(io_err, Cint[n], Cint[m], x, y, ne, Cint[lhe_ptr],
    he_row_ptr_cp, he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val],
    he_val, Cint[byrows])
  @cutest_error
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return ne[1]
end

function ceh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    lhe_ptr::Int, he_row_ptr::Array{Cint, 1}, he_val_ptr::Array{Cint, 1},
    lhe_row::Int, he_row::Array{Cint, 1}, lhe_val::Int,
    he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  ne = Cint[0]
  ceh(io_err, Cint[n], Cint[m], x, y, ne, Cint[lhe_ptr], he_row_ptr,
    he_val_ptr, Cint[lhe_row], he_row, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  return ne[1]
end

"""
    h = cidh(n, x, iprob, lh1)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function cidh(n::Int, x::Array{Float64, 1}, iprob::Int, lh1::Int)
  io_err = Cint[0]
  h = Array(Cdouble, lh1, n)
  cidh(io_err, Cint[n], x, Cint[iprob], Cint[lh1], h)
  @cutest_error
  return h
end

"""
    cidh!(n, x, iprob, lh1, h)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function cidh!(n::Int, x::Array{Float64, 1}, iprob::Int, lh1::Int, h::Array{Float64,
    2})
  io_err = Cint[0]
  cidh(io_err, Cint[n], x, Cint[iprob], Cint[lh1], h)
  @cutest_error
  return
end

"""
    h = cidh(nlp, x, iprob, lh1)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function cidh(nlp::CUTEstModel, x::Array{Float64, 1}, iprob::Int, lh1::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  h = Array(Cdouble, lh1, n)
  cidh(io_err, Cint[n], x, Cint[iprob], Cint[lh1], h)
  @cutest_error
  return h
end

"""
    cidh!(nlp, x, iprob, lh1, h)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - lh1:     [IN] Int
  - h:       [OUT] Array{Float64, 2}
"""
function cidh!(nlp::CUTEstModel, x::Array{Float64, 1}, iprob::Int, lh1::Int,
    h::Array{Float64, 2})
  io_err = Cint[0]
  n = nlp.meta.nvar
  cidh(io_err, Cint[n], x, Cint[iprob], Cint[lh1], h)
  @cutest_error
  return
end

"""
    nnzh, h_val, h_row, h_col = cish(n, x, iprob, lh)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cish(n::Int, x::Array{Float64, 1}, iprob::Int, lh::Int)
  io_err = Cint[0]
  nnzh = Cint[0]
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  cish(io_err, Cint[n], x, Cint[iprob], nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = cish!(n, x, iprob, lh, h_val, h_row, h_col)

  - n:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cish!(n::Int, x::Array{Float64, 1}, iprob::Int, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  nnzh = Cint[0]
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  cish(io_err, Cint[n], x, Cint[iprob], nnzh, Cint[lh], h_val,
    h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function cish!(n::Int, x::Array{Float64, 1}, iprob::Int, lh::Int,
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  nnzh = Cint[0]
  cish(io_err, Cint[n], x, Cint[iprob], nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1]
end

"""
    nnzh, h_val, h_row, h_col = cish(nlp, x, iprob)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cish(nlp::CUTEstModel, x::Array{Float64, 1}, iprob::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  cish(io_err, Cint[n], x, Cint[iprob], nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1], h_val, h_row, h_col
end

"""
    nnzh = cish!(nlp, x, iprob, h_val, h_row, h_col)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - iprob:   [IN] Int
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function cish!(nlp::CUTEstModel, x::Array{Float64, 1}, iprob::Int,
    h_val::Array{Float64, 1}, h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  cish(io_err, Cint[n], x, Cint[iprob], nnzh, Cint[lh], h_val,
    h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzh[1]
end

function cish!(nlp::CUTEstModel, x::Array{Float64, 1}, iprob::Int,
    h_val::Array{Float64, 1}, h_row::Array{Cint, 1}, h_col::Array{Cint,
    1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  cish(io_err, Cint[n], x, Cint[iprob], nnzh, Cint[lh], h_val, h_row,
    h_col)
  @cutest_error
  return nnzh[1]
end

"""
    nnzj, j_val, j_var, j_fun, nnzh, h_val, h_row, h_col = csgrsh(n, m, x, y, grlagf, lj, lh)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - lj:      [IN] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csgrsh(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, lh::Int)
  io_err = Cint[0]
  nnzj = Cint[0]
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  nnzh = Cint[0]
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  csgrsh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzj[1], j_val, j_var, j_fun, nnzh[1], h_val, h_row, h_col
end

"""
    nnzj, nnzh = csgrsh!(n, m, x, y, grlagf, lj, j_val, j_var, j_fun, lh, h_val, h_row, h_col)

  - n:       [IN] Int
  - m:       [IN] Int
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - lj:      [IN] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - nnzh:    [OUT] Int
  - lh:      [IN] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csgrsh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, j_val::Array{Float64, 1}, j_var::Array{Int, 1},
    j_fun::Array{Int, 1}, lh::Int, h_val::Array{Float64, 1},
    h_row::Array{Int, 1}, h_col::Array{Int, 1})
  io_err = Cint[0]
  nnzj = Cint[0]
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  nnzh = Cint[0]
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  csgrsh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var_cp, j_fun_cp, nnzh, Cint[lh], h_val, h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzj[1], nnzh[1]
end

function csgrsh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, j_val::Array{Float64, 1}, j_var::Array{Cint,
    1}, j_fun::Array{Cint, 1}, lh::Int, h_val::Array{Float64, 1},
    h_row::Array{Cint, 1}, h_col::Array{Cint, 1})
  io_err = Cint[0]
  nnzj = Cint[0]
  nnzh = Cint[0]
  csgrsh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzj[1], nnzh[1]
end

"""
    nnzj, j_val, j_var, j_fun, nnzh, h_val, h_row, h_col = csgrsh(nlp, x, y, grlagf)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csgrsh(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj + nlp.meta.nvar
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_val = Array(Cdouble, lh)
  h_row = Array(Cint, lh)
  h_col = Array(Cint, lh)
  csgrsh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzj[1], j_val, j_var, j_fun, nnzh[1], h_val, h_row, h_col
end

"""
    nnzj, nnzh = csgrsh!(nlp, x, y, grlagf, j_val, j_var, j_fun, h_val, h_row, h_col)

  - nlp:     [IN] CUTEstModel
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - grlagf:  [IN] Bool
  - nnzj:    [OUT] Int
  - j_val:   [OUT] Array{Float64, 1}
  - j_var:   [OUT] Array{Int, 1}
  - j_fun:   [OUT] Array{Int, 1}
  - nnzh:    [OUT] Int
  - h_val:   [OUT] Array{Float64, 1}
  - h_row:   [OUT] Array{Int, 1}
  - h_col:   [OUT] Array{Int, 1}
"""
function csgrsh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, j_val::Array{Float64, 1}, j_var::Array{Int, 1},
    j_fun::Array{Int, 1}, h_val::Array{Float64, 1}, h_row::Array{Int, 1},
    h_col::Array{Int, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj + nlp.meta.nvar
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  h_row_cp = Array(Cint, lh)
  h_col_cp = Array(Cint, lh)
  csgrsh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var_cp, j_fun_cp, nnzh, Cint[lh], h_val, h_row_cp, h_col_cp)
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  for i = 1:lh
    h_row[i] = h_row_cp[i]
  end
  for i = 1:lh
    h_col[i] = h_col_cp[i]
  end
  return nnzj[1], nnzh[1]
end

function csgrsh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, j_val::Array{Float64, 1}, j_var::Array{Cint, 1},
    j_fun::Array{Cint, 1}, h_val::Array{Float64, 1}, h_row::Array{Cint,
    1}, h_col::Array{Cint, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj + nlp.meta.nvar
  nnzh = Cint[0]
  lh = nlp.meta.nnzh
  csgrsh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, nnzh, Cint[lh], h_val, h_row, h_col)
  @cutest_error
  return nnzj[1], nnzh[1]
end

"""
    nnzj, j_val, j_var, j_fun, ne, he_row_ptr, he_val_ptr, he_row, he_val = csgreh(n, m, x, y, grlagf, lj, lhe_ptr, lhe_row, lhe_val, byrows)

  - n:          [IN] Int
  - m:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - grlagf:     [IN] Bool
  - nnzj:       [OUT] Int
  - lj:         [IN] Int
  - j_val:      [OUT] Array{Float64, 1}
  - j_var:      [OUT] Array{Int, 1}
  - j_fun:      [OUT] Array{Int, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function csgreh(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, lhe_ptr::Int, lhe_row::Int, lhe_val::Int,
    byrows::Bool)
  io_err = Cint[0]
  nnzj = Cint[0]
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  csgreh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return nnzj[1], j_val, j_var, j_fun, ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    nnzj, ne = csgreh!(n, m, x, y, grlagf, lj, j_val, j_var, j_fun, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - n:          [IN] Int
  - m:          [IN] Int
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - grlagf:     [IN] Bool
  - nnzj:       [OUT] Int
  - lj:         [IN] Int
  - j_val:      [OUT] Array{Float64, 1}
  - j_var:      [OUT] Array{Int, 1}
  - j_fun:      [OUT] Array{Int, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function csgreh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, j_val::Array{Float64, 1}, j_var::Array{Int, 1},
    j_fun::Array{Int, 1}, lhe_ptr::Int, he_row_ptr::Array{Int, 1},
    he_val_ptr::Array{Int, 1}, lhe_row::Int, he_row::Array{Int, 1},
    lhe_val::Int, he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  nnzj = Cint[0]
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  csgreh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var_cp, j_fun_cp, ne, Cint[lhe_ptr], he_row_ptr_cp,
    he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return nnzj[1], ne[1]
end

function csgreh!(n::Int, m::Int, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lj::Int, j_val::Array{Float64, 1}, j_var::Array{Cint,
    1}, j_fun::Array{Cint, 1}, lhe_ptr::Int, he_row_ptr::Array{Cint, 1},
    he_val_ptr::Array{Cint, 1}, lhe_row::Int, he_row::Array{Cint, 1},
    lhe_val::Int, he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  nnzj = Cint[0]
  ne = Cint[0]
  csgreh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return nnzj[1], ne[1]
end

"""
    nnzj, j_val, j_var, j_fun, ne, he_row_ptr, he_val_ptr, he_row, he_val = csgreh(nlp, x, y, grlagf, lhe_ptr, lhe_row, lhe_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - grlagf:     [IN] Bool
  - nnzj:       [OUT] Int
  - j_val:      [OUT] Array{Float64, 1}
  - j_var:      [OUT] Array{Int, 1}
  - j_fun:      [OUT] Array{Int, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function csgreh(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, lhe_ptr::Int, lhe_row::Int, lhe_val::Int, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj
  j_val = Array(Cdouble, lj)
  j_var = Array(Cint, lj)
  j_fun = Array(Cint, lj)
  ne = Cint[0]
  he_row_ptr = Array(Cint, lhe_ptr)
  he_val_ptr = Array(Cint, lhe_ptr)
  he_row = Array(Cint, lhe_row)
  he_val = Array(Cdouble, lhe_val)
  csgreh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return nnzj[1], j_val, j_var, j_fun, ne[1], he_row_ptr, he_val_ptr, he_row, he_val
end

"""
    nnzj, ne = csgreh!(nlp, x, y, grlagf, j_val, j_var, j_fun, lhe_ptr, he_row_ptr, he_val_ptr, lhe_row, he_row, lhe_val, he_val, byrows)

  - nlp:        [IN] CUTEstModel
  - x:          [IN] Array{Float64, 1}
  - y:          [IN] Array{Float64, 1}
  - grlagf:     [IN] Bool
  - nnzj:       [OUT] Int
  - j_val:      [OUT] Array{Float64, 1}
  - j_var:      [OUT] Array{Int, 1}
  - j_fun:      [OUT] Array{Int, 1}
  - ne:         [OUT] Int
  - lhe_ptr:    [IN] Int
  - he_row_ptr: [OUT] Array{Int, 1}
  - he_val_ptr: [OUT] Array{Int, 1}
  - lhe_row:    [IN] Int
  - he_row:     [OUT] Array{Int, 1}
  - lhe_val:    [IN] Int
  - he_val:     [OUT] Array{Float64, 1}
  - byrows:     [IN] Bool
"""
function csgreh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, j_val::Array{Float64, 1}, j_var::Array{Int, 1},
    j_fun::Array{Int, 1}, lhe_ptr::Int, he_row_ptr::Array{Int, 1},
    he_val_ptr::Array{Int, 1}, lhe_row::Int, he_row::Array{Int, 1},
    lhe_val::Int, he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj
  j_var_cp = Array(Cint, lj)
  j_fun_cp = Array(Cint, lj)
  ne = Cint[0]
  he_row_ptr_cp = Array(Cint, lhe_ptr)
  he_val_ptr_cp = Array(Cint, lhe_ptr)
  he_row_cp = Array(Cint, lhe_row)
  csgreh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var_cp, j_fun_cp, ne, Cint[lhe_ptr], he_row_ptr_cp,
    he_val_ptr_cp, Cint[lhe_row], he_row_cp, Cint[lhe_val], he_val,
    Cint[byrows])
  @cutest_error
  for i = 1:lj
    j_var[i] = j_var_cp[i]
  end
  for i = 1:lj
    j_fun[i] = j_fun_cp[i]
  end
  for i = 1:lhe_ptr
    he_row_ptr[i] = he_row_ptr_cp[i]
  end
  for i = 1:lhe_ptr
    he_val_ptr[i] = he_val_ptr_cp[i]
  end
  for i = 1:lhe_row
    he_row[i] = he_row_cp[i]
  end
  return nnzj[1], ne[1]
end

function csgreh!(nlp::CUTEstModel, x::Array{Float64, 1}, y::Array{Float64, 1},
    grlagf::Bool, j_val::Array{Float64, 1}, j_var::Array{Cint, 1},
    j_fun::Array{Cint, 1}, lhe_ptr::Int, he_row_ptr::Array{Cint, 1},
    he_val_ptr::Array{Cint, 1}, lhe_row::Int, he_row::Array{Cint, 1},
    lhe_val::Int, he_val::Array{Float64, 1}, byrows::Bool)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  nnzj = Cint[0]
  lj = nlp.meta.nnzj
  ne = Cint[0]
  csgreh(io_err, Cint[n], Cint[m], x, y, Cint[grlagf], nnzj, Cint[lj],
    j_val, j_var, j_fun, ne, Cint[lhe_ptr], he_row_ptr, he_val_ptr,
    Cint[lhe_row], he_row, Cint[lhe_val], he_val, Cint[byrows])
  @cutest_error
  return nnzj[1], ne[1]
end

"""
    result = chprod(n, m, goth, x, y, vector)

  - n:       [IN] Int
  - m:       [IN] Int
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chprod(n::Int, m::Int, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1})
  io_err = Cint[0]
  result = Array(Cdouble, n)
  chprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return result
end

"""
    chprod!(n, m, goth, x, y, vector, result)

  - n:       [IN] Int
  - m:       [IN] Int
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chprod!(n::Int, m::Int, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1}, result::Array{Float64, 1})
  io_err = Cint[0]
  chprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return
end

"""
    result = chprod(nlp, goth, x, y, vector)

  - nlp:     [IN] CUTEstModel
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chprod(nlp::CUTEstModel, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  result = Array(Cdouble, n)
  chprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return result
end

"""
    chprod!(nlp, goth, x, y, vector, result)

  - nlp:     [IN] CUTEstModel
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chprod!(nlp::CUTEstModel, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1}, result::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  chprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return
end

"""
    result = chcprod(n, m, goth, x, y, vector)

  - n:       [IN] Int
  - m:       [IN] Int
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chcprod(n::Int, m::Int, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1})
  io_err = Cint[0]
  result = Array(Cdouble, n)
  chcprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return result
end

"""
    chcprod!(n, m, goth, x, y, vector, result)

  - n:       [IN] Int
  - m:       [IN] Int
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chcprod!(n::Int, m::Int, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1}, result::Array{Float64, 1})
  io_err = Cint[0]
  chcprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return
end

"""
    result = chcprod(nlp, goth, x, y, vector)

  - nlp:     [IN] CUTEstModel
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chcprod(nlp::CUTEstModel, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  result = Array(Cdouble, n)
  chcprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return result
end

"""
    chcprod!(nlp, goth, x, y, vector, result)

  - nlp:     [IN] CUTEstModel
  - goth:    [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - y:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - result:  [OUT] Array{Float64, 1}
"""
function chcprod!(nlp::CUTEstModel, goth::Bool, x::Array{Float64, 1}, y::Array{Float64,
    1}, vector::Array{Float64, 1}, result::Array{Float64, 1})
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  chcprod(io_err, Cint[n], Cint[m], Cint[goth], x, y, vector, result)
  @cutest_error
  return
end

"""
    result = cjprod(n, m, gotj, jtrans, x, vector, lvector, lresult)

  - n:       [IN] Int
  - m:       [IN] Int
  - gotj:    [IN] Bool
  - jtrans:  [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - lvector: [IN] Int
  - result:  [OUT] Array{Float64, 1}
  - lresult: [IN] Int
"""
function cjprod(n::Int, m::Int, gotj::Bool, jtrans::Bool, x::Array{Float64, 1},
    vector::Array{Float64, 1}, lvector::Int, lresult::Int)
  io_err = Cint[0]
  result = Array(Cdouble, lresult)
  cjprod(io_err, Cint[n], Cint[m], Cint[gotj], Cint[jtrans], x,
    vector, Cint[lvector], result, Cint[lresult])
  @cutest_error
  return result
end

"""
    cjprod!(n, m, gotj, jtrans, x, vector, lvector, result, lresult)

  - n:       [IN] Int
  - m:       [IN] Int
  - gotj:    [IN] Bool
  - jtrans:  [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - lvector: [IN] Int
  - result:  [OUT] Array{Float64, 1}
  - lresult: [IN] Int
"""
function cjprod!(n::Int, m::Int, gotj::Bool, jtrans::Bool, x::Array{Float64, 1},
    vector::Array{Float64, 1}, lvector::Int, result::Array{Float64, 1},
    lresult::Int)
  io_err = Cint[0]
  cjprod(io_err, Cint[n], Cint[m], Cint[gotj], Cint[jtrans], x,
    vector, Cint[lvector], result, Cint[lresult])
  @cutest_error
  return
end

"""
    result = cjprod(nlp, gotj, jtrans, x, vector, lvector, lresult)

  - nlp:     [IN] CUTEstModel
  - gotj:    [IN] Bool
  - jtrans:  [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - lvector: [IN] Int
  - result:  [OUT] Array{Float64, 1}
  - lresult: [IN] Int
"""
function cjprod(nlp::CUTEstModel, gotj::Bool, jtrans::Bool, x::Array{Float64, 1},
    vector::Array{Float64, 1}, lvector::Int, lresult::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  result = Array(Cdouble, lresult)
  cjprod(io_err, Cint[n], Cint[m], Cint[gotj], Cint[jtrans], x,
    vector, Cint[lvector], result, Cint[lresult])
  @cutest_error
  return result
end

"""
    cjprod!(nlp, gotj, jtrans, x, vector, lvector, result, lresult)

  - nlp:     [IN] CUTEstModel
  - gotj:    [IN] Bool
  - jtrans:  [IN] Bool
  - x:       [IN] Array{Float64, 1}
  - vector:  [IN] Array{Float64, 1}
  - lvector: [IN] Int
  - result:  [OUT] Array{Float64, 1}
  - lresult: [IN] Int
"""
function cjprod!(nlp::CUTEstModel, gotj::Bool, jtrans::Bool, x::Array{Float64, 1},
    vector::Array{Float64, 1}, lvector::Int, result::Array{Float64, 1},
    lresult::Int)
  io_err = Cint[0]
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  cjprod(io_err, Cint[n], Cint[m], Cint[gotj], Cint[jtrans], x,
    vector, Cint[lvector], result, Cint[lresult])
  @cutest_error
  return
end

"""
    uterminate()

"""
function uterminate()
  io_err = Cint[0]
  uterminate(io_err)
  @cutest_error
  return
end

"""
    cterminate()

"""
function cterminate()
  io_err = Cint[0]
  cterminate(io_err)
  @cutest_error
  return
end


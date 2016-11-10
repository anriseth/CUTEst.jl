# Tests made to increase the coverage.

io_err = Cint[0]

function coverage_increase(nlp :: CUTEstModel)
  n, m = nlp.meta.nvar, nlp.meta.ncon
  pname = Array{UInt8, 1}(10)
  probname(nlp.lib, io_err, pname)
  vname = Array{UInt8, 2}(10, n)
  varnames(nlp.lib, io_err, Cint[n], vname)
  if m == 0
    unames(nlp.lib, io_err, Cint[n], pname, vname)
    calls, time = Array{Cdouble}(4), Array{Cdouble}(2)
    ureport(nlp.lib, io_err, calls, time)
  else
    cname = Array{UInt8, 2}(10, m)
    connames(nlp.lib, io_err, Cint[m], cname)
    cnames(nlp.lib, io_err, Cint[n], Cint[m], pname, vname, cname)
    calls, time = Array{Cdouble}(7), Array{Cdouble}(2)
    creport(nlp.lib, io_err, calls, time)
    nvo, nvc, ec, lc = Cint[0], Cint[0], Cint[0], Cint[0]
    cstats(nlp.lib, io_err, nvo, nvc, ec, lc)

    lchp = Cint[0]
    cdimchp(nlp.lib, io_err, lchp)
    chp_ind, chp_ptr = Array{Cint}(lchp[1]), Array{Cint}(m+1)
    cchprodsp(nlp.lib, io_err, Cint[m], lchp, chp_ind, chp_ptr)
    lj, nnzj = Cint[0], Cint[0]
    cdimsj(nlp.lib, io_err, lj)
    j_var, j_fun = Array{Cint}(lj[1]), Array{Cint}(lj[1])
    csgrp(nlp.lib, io_err, Cint[n], nnzj, lj, j_var, j_fun)
    lh, nnzh = Cint[0], Cint[0]
    cdimsh(nlp.lib, io_err, lh)
    h_row, h_col = Array{Cint}(lh[1]), Array{Cint}(lh[1])
    csgrshp(nlp.lib, io_err, Cint[n], nnzj, lj, j_var, j_fun, nnzh, lh, h_row, h_col)
  end
end

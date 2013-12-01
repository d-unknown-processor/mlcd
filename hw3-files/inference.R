mean.field.inference <- function()
{
  q.a <- new.joint(1)
  q.b <- new.joint(1)
  q.c <- new.joint(1)
  q.d <- new.joint(1)
  q.e <- new.joint(1)
  q.f <- new.joint(1)

  a.up <- function(a)
  {
    numerator = 0
    # Sum over the values of B
    for (i in 1:2 ) {
      numerator = numerator + q.b[i]*log(prob.b.given.a(i,a)*prob.a(a))
    }
    
    f_ac <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (a in 1:2) {
        for (b in 1:2) {
          f_ac[a,c] = f_ac[a,c] +  prob.c.given.ab(c,a,b)
        }
      }
    }
    for (c in 1:2 ) {
      numerator = numerator + q.c[c]*log(f_ac[a,c])
    }
    exp(numerator)
  }

  b.up <- function(b)
  {
    numerator = 0
    # Sum over values of a
    for (i in 1:2 ) {
      numerator = numerator + q.a[i]*log(prob.b.given.a(b,i)*prob.a(a))
    }
 
   
    f_bc <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (a in 1:2) {
        for (b in 1:2) {
          f_bc[b,c] = f_bc[b,c] +  prob.c.given.ab(c,a,b)
        }
      }
    }
    # Sum over the values of c, marginalize out a
    for (c in 1:2 ) {
      numerator = numerator + q.c[c]*log(f_bc[b,c])
    }
    # Sum over values of d
    for (i in 1:2 ) {
      numerator = numerator + q.d[i]*log(prob.d.given.b(i,b))
    }
    
    exp(numerator)
  }

  c.up <- function(c)
  {
    numerator = 0
    f_cb <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (a in 1:2) {
        for (b in 1:2) {
          f_cb[c,b] = f_cb[c,b] +  prob.c.given.ab(c,a,b)
        }
      }
    }
    f_ca <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (a in 1:2) {
        for (b in 1:2) {
          f_ca[c,a] = f_ca[c,a] +  prob.c.given.ab(c,a,b)
        }
      }
    }
    f_ce <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (e in 1:2) {
        for (d in 1:2) {
          f_ce[c,e] = f_ce[c,e] +  prob.e.given.cd(e,c,d)
        }
      }
    }
    f_cd <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (e in 1:2) {
        for (d in 1:2) {
          f_cd[c,d] = f_cd[c,d] +  prob.e.given.cd(e,c,d)
        }
      }
    }
    # Sum over values of a
    for (a in 1:2 ) {
      numerator = numerator + q.a[a]*log(f_ca[c,a])
    }
    # Sum over the values of b
    for (b in 1:2 ) {
      numerator = numerator + q.b[b]*log(f_cb[c,b])
    }
    
    # Sum over the values of e
    for (e in 1:2 ) {
      numerator = numerator + q.e[e]*log(f_ce[c,e])
    }

    # Sum over the values of d
    for (d in 1:2 ) {
      numerator = numerator + q.d[d]*log(f_cd[c,d])
    }
    
    exp(numerator)
  }

  d.up <- function(d)
  {
    f_de <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (e in 1:2) {
        for (d in 1:2) {
          f_de[d,e] = f_de[d,e] +  prob.e.given.cd(e,c,d)
        }
      }
    }
    f_dc <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (e in 1:2) {
        for (d in 1:2) {
          f_dc[d,c] = f_dc[d,c] +  prob.e.given.cd(e,c,d)
        }
      }
    }
    numerator = 0
    # Sum over values of b
    for (i in 1:2 ) {
      numerator = numerator + q.b[i]*log(prob.d.given.b(d,i))
    }
    # Sum over the values of f
    for (i in 1:2 ) {
      numerator = numerator + q.f[i]*log(prob.f.given.d(i,d))
    }
    # Sum over the values of e
    for (e in 1:2 ) {
      numerator = numerator + q.e[e]*log(f_de[d,e])
    }
    # Sum over the values of c
    for (c in 1:2 ) {
      numerator = numerator + q.c[c]*log(f_dc[d,c])
    }
    exp(numerator)
  }

  e.up <- function(e)
  {
    numerator = 0
    f_ed <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (e in 1:2) {
        for (d in 1:2) {
          f_ed[e,d] = f_ed[e,d] +  prob.e.given.cd(e,c,d)
        }
      }
    }
    f_ec <- array(0, dim=c(2,2))
    for (c in 1:2) {
      for (e in 1:2) {
        for (d in 1:2) {
          f_ec[e,c] = f_ec[e,c] + prob.e.given.cd(e,c,d)
        }
      }
    }
    # Sum over values of d, marginalize out c
    for (d in 1:2 ) {
      numerator = numerator + q.d[d]*log(f_ed[e,d])
    }
    # Sum over values of c, marginalize out d
    for (c in 1:2 ) {
      numerator = numerator + q.c[c]*log(f_ec[e,c])
    }
    exp(numerator)
  }

  f.up <- function(f)
  {
    numerator = 0
    # Sum over the values of d
    for (i in 1:2 ) {
      numerator = numerator + q.d[i]*log(prob.f.given.d(f,i))
    }
    exp(numerator)
  }

  niter <- 0
  converged <- FALSE
  tol <- 1e-3

  close.enough <- function(q, q.old)
  {
    max(abs(q - q.old)) < tol
  }

  while (!converged)
  {
    q.a.old <- q.a
    q.b.old <- q.b
    q.c.old <- q.c
    q.d.old <- q.d
    q.e.old <- q.e
    q.f.old <- q.f

    q.a <- c(a.up(1), a.up(2)) / sum(a.up(1), a.up(2))
    q.b <- c(b.up(1), b.up(2)) / sum(b.up(1), b.up(2))
    q.c <- c(c.up(1), c.up(2)) / sum(c.up(1), c.up(2))
    q.d <- c(d.up(1), d.up(2)) / sum(d.up(1), d.up(2))
    q.e <- c(e.up(1), e.up(2)) / sum(e.up(1), e.up(2))
    q.f <- c(f.up(1), f.up(2)) / sum(f.up(1), f.up(2))

    niter <- niter + 1

    converged <- all(close.enough(q.a, q.a.old),
                     close.enough(q.b, q.b.old),
                     close.enough(q.c, q.c.old),
                     close.enough(q.d, q.d.old),
                     close.enough(q.e, q.e.old),
                     close.enough(q.f, q.f.old))
    print(converged)
  }

  q.full <- function(a, b, c, d, e, f)
  {
    q.a[a] * q.b[b] * q.c[c] * q.d[d] * q.e[e] * q.f[f]
  }

  make.full.joint(q.full)
}

struct.mean.field.inference <- function()
{
  q.abc <- new.joint(3)
  q.def <- new.joint(3)

  q.b <-  function(b)    sum(q.abc[, b, ])
  q.c <-  function(c)    sum(q.abc[, , c])
  q.d <-  function(d)    sum(q.def[d, , ])
  q.de <- function(d, e) sum(q.def[d, e, ])

  abc.up <- function(a, b, c)
  {
    stop("You need to implement the update for Q(ABC)")
  }

  def.up <- function(d, e, f)
  {
    stop("You need to implement the update for Q(DEF)")
  }

  niter <- 0
  converged <- FALSE
  tol <- 1e-3

  close.enough <- function(q, q.old)
  {
    max(abs(q - q.old)) < tol
  }

  while (!converged)
  {
    q.abc.old <- q.abc
    q.def.old <- q.def

    combs <- as.matrix(expand.grid(1:2, 1:2, 1:2))

    ## Update A,B,C

    for (i in seq_len(nrow(combs)))
    {
      a <- combs[i, 1]
      b <- combs[i, 2]
      c <- combs[i, 3]

      q.abc[a, b, c] <- abc.up(a, b, c)
    }

    q.abc <- q.abc / sum(q.abc)

    ## Update D,E,F

    for (i in seq_len(nrow(combs)))
    {
      d <- combs[i, 1]
      e <- combs[i, 2]
      f <- combs[i, 3]

      q.def[d, e, f] <- def.up(d, e, f)
    }

    q.def <- q.def / sum(q.def)

    niter <- niter + 1

    converged <- all(close.enough(q.abc, q.abc.old),
                     close.enough(q.def, q.def.old))
  }

  q.full <- function(a, b, c, d, e, f)
  {
    q.abc[a, b, c] * q.def[d, e, f]
  }

  make.full.joint(q.full)
}

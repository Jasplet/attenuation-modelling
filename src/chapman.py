#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:50:54 2022

@author: ja17375
"""

import numpy as np


def chapmab_anic():
    '''
    Uses Chapman (2003) modelling to calculate the frequency dependent anisotropic tensor for a cracked solid.
    '''
    pass


### MATLAB TO ADAPT

# function [c11, c33, c12, c13, c44] = anic(q,lam, mu,ar, fd, af, cd, por, etaw, etag, Kw, Kg, sw,tau, w) 
# %  parameters q,lam, mu, fd, af, cd, por, etaw, etag, Kw, Kg, sw, w0, w 
# %  This subroutine takes real values for the lamee constants (lam & mu), the density (rho) fracture 
# %  density (fd), fracture length in meters (af), the microcrack density (cd), the porosity (ppor)
# %  the parameter gamma (gam) related to fluid compressibility, the time constant (tau) related to the
# %  inverse of the squirt flow frequency and the wave frequency (w). 

# %  The output is a complex elastic tensor, with the x_3 axis assumed to be the axis of symmetry. This 
# %  elastic tensor is characterised by the constants c11, c33, c12, c13 and c44.

# grainsize=120e-6;
# i=complex(0,1);
# pr=lam/(2*lam+2*mu); % poissons ratio
# sigc=pi*mu*ar/(2-2*pr); % sigma of a crack

# cpor=4/3*pi*cd*ar; % crack porosity
# fpor=4/3*pi*fd*ar; % fracture porosity
# ppor=por-cpor-fpor; % pore porosity

# % relative permeability model
# kapw=sw^3; % relative permeability of water
# kapg=(1-sw)^2; % relative permeability of gas

# % patch parameter q
# qp=sw+q*(1-sw);

# % Kf the modified fluid bulk modulus, an average of two fluid moduli
# % weighted by q
# kf=qp/(sw/Kw+q*(1-sw)/Kg);
# kp=4*mu/(3*kf);
# kc=sigc/kf;

# % parameter values
# eta=qp/(kapw/etaw+q*kapg/etag); % new mixed viscosity
# p0=(1+sigc/Kw)*etaw/tau; % tau is grain-scale value at full water saturation
# taum=(1+kc)*eta/p0; % grain-scale tau (partial saturation)
# tauf=af*taum/grainsize; % fracture-scale tau (partial saturaiton)

# % some definitions
# gamma=3*pi*(1+kp)/(8*(1-pr)*(1+kc));
# gammap=gamma*(1-pr)/(1+pr)/(1+kp);

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                             

#        l2=lam*lam + 4/3*lam*mu +0.8*mu*mu;

#        l3=4*(lam*lam + 4/3*lam*mu + 8/15*mu*mu );


#        kap=lam+ 2/3*mu;
       

#        io=(cpor/ar)/( (cpor/ar) + ppor);
 
#        beta=(fpor/ar)/((cpor/ar) + ppor);


# % calculate d1 and d2, f1 and f2

#        z1=(io/(3 + 3*kc)) + (gammap - gammap*io);
#        z2=((i*w*taum)*( 1/(3 + 3*kc) -gammap))/(1+i*w*taum);
#        z3=io +( (io*beta)/(1+i*w*tauf));

#        a=z1 - (z2*z3);

#        b=beta/( (1+kc)*(1+i*w*tauf) );


#        z4=((1-io)*gamma) + (( (1-io)*beta)/(1+i*w*tauf));
#        z5=(io + ((io*beta)/(1 + i*w*tauf)))*((1+i*w*gamma*taum)/(1+i*w*taum));
       
#        c= z4 + z5;



#        d1=a/c;



#        d2=b/c;




#        x1=io*d1*(1+i*w*gamma*taum)/(1+i*w*taum);
#        x2= d1*(1-io);
#        x3=(((1/(3 + 3*kc)) -gammap)*io*i*w*taum)/(1+i*w*taum);

#        f1=(x1 + x2 + x3)/(1 + i*w*tauf);



#        v1=(i*w*tauf)/(1 + kc);
#        v2=(io*d2*(1+i*w*gamma*taum))/(1+i*w*taum);
#        v3=d2*(1-io);


#        f2=(v1+v2+v3)/(1+i*w*tauf);


# % end definiton of d1, d2, f1, f2

# % define g1, g2, g3


      
#        g1=(i*w*taum)/( (1+i*w*taum)*(1+kc));

#        g2=(d1 +(d1*i*w*taum*gamma) - (gammap*i*w*taum))/(1+i*w*taum);

#        g3=(d2*(1+i*w*gamma*taum))/(1+i*w*taum);



# % end of definition of g1, g2, g3

# % calculation of c_11


#        aa1=(l2/sigc) + ((32*(1-pr)*mu)/(15*(2-pr)*pi*ar));

#        aa2=(((l2/sigc)+kap)*g1)+((((3*kap*kap)/sigc) +3*kap)*g2)+( (((lam*kap)/sigc)+lam)*g3);


# % crack correction cpor*(aa1-aa2)



#        aa3=( (3*lam*lam + 4*lam*mu+ (mu*mu*(36+20*pr)/(7-5*pr)) )*3*(1-pr))/(4*(1+pr)*mu);

#        aa4=(1+(3*kap)/(4*mu))*(3*kap*d1 + lam*d2);



# % porosity correction ppor*(aa3-aa4)

#        aa5=lam*lam/sigc;

#        aa6=((3*lam*kap/sigc)+(3*kap))*f1 + ( ( (lam*lam/sigc) + lam)*f2);

# % fracture correction fpor*(aa5-aa6)

#        c11= (lam+2*mu) + ((-1)*cpor*(aa1-aa2)) + ((-1)*ppor*(aa3-aa4)) + ((-1)*fpor*(aa5-aa6));



# % calculation of c33


       
      

#        bb2=(((l2/sigc)+kap)*g1)+(( ((3*kap*kap)/sigc) +3*kap)*g2)+(((((lam+2*mu)*kap)/sigc)+(lam+2*mu))*g3);


# % crack correction cpor*(aa1-bb2)



#        bb3=(1+((3*kap)/(4*mu)))*( (3*kap*d1) + ((lam+2*mu)*d2) );

       

# % porosity correction ppor*(aa3-bb3)

#        bb4=(lam +2*mu)*(lam+2*mu)/sigc;

#        bb5=(( (3*kap*(lam+2*mu)/sigc) +3*kap)*f1) + f2*( ((lam+2*mu)*(lam+2*mu)/sigc) + lam + 2*mu);

       

# % fracture correction fpor*(bb4-bb5)


#        c33=(lam+2*mu) + ((-1)*cpor*(aa1-bb2)) + ((-1)*ppor*(aa3-bb3)) +((-1)*fpor*(bb4-bb5));


# % calculation of c44


#        jj1=(0.2666666666667)*mu*mu*(1-g1)/sigc;

#        jj2=((1.6)*(1-pr)*mu)/(pi*ar*(2-pr));

# % crack correction cpor*(jj1 + jj2)



#        jj3=(15*(1-pr)*mu)/(7-5*pr);

# % porosity correction ppor*jj3


#        jj4=(4*(1-pr)*mu)/(pi*(2-pr)*ar);

# % fracture correction fpor*jj4


#        c44= mu + ((-1)*cpor*(jj1+jj2)) + ((-1)*ppor*jj3) + ((-1)*fpor*jj4);


# % calculation of c12

#        kk1=(l3/sigc) + ((32*(1-pr)*mu)/(15*(2-pr)*pi*ar));

#        kk2=(((l3/sigc) + 4*kap)*g1) + (((12*kap*kap/sigc) + 12*kap)*g2) + (((4*lam*kap/sigc) + 4*lam)*g3);

# % crack correction cpor*(kk1-kk2)

#        kk3= ((12*lam*lam + 16*lam*mu +(mu*mu*64/(7-5*pr)))*3*(1-pr))/(4*(1+pr)*mu) ;


#        kk4=( (1.5*kap/mu) + 2)*(6*kap*d1 + 2*lam*d2);


# % porosity correction ppor*(kk3-kk4)


#        kk5=4*lam*lam/sigc;

#        kk6=(f1*( (12*kap*lam/sigc) + 12*kap))+ (f2*( (4*lam*lam/sigc) + 4*lam));

# % fracture correction fpor*(kk5-kk6)


#        c12=0.5*( (4*lam+4*mu) + ((-2)*c11) + ((-1)*cpor*(kk1-kk2)) + ((-1)*ppor *(kk3-kk4)) + ((-1)*fpor*(kk5-kk6)) );



# % calculation of c13

#        mm2=(((l3/sigc) + 4*kap)*g1) +(((12*kap*kap/sigc) + 12*kap)*g2)+(((4*(lam+mu) *kap/sigc) + 4*(lam+mu))*g3);
 
# % crack correction cpor*(kk1-mm2)

#        mm4=( (1.5*kap/mu) + 2)*(6*kap*d1 + (2*lam+2*mu)*d2);

# % porosity correction ppor*(kk3-mm4)

#        mm5=(4*(lam+mu)*(lam+mu))/sigc;

#        mm6=(f1*( (12*kap*(lam+mu)/sigc) + 12*kap)) + (f2*( (4*(lam+mu)*(lam+mu)/sigc)+ 4*(lam+mu)));

# % fracture correction fpor*(mm5-mm6)


#        c13=0.5*( (4*lam+4*mu) + ((-1)*(c11 + c33)) + ((-1)*cpor*(kk1-mm2)) + ((-1)*ppor*(kk3-mm4)) +((-1)*fpor*(mm5-mm6)) );
       
# end     
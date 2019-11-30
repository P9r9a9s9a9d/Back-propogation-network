# Input for Back propogation
# Binary sigmoidal activation function

x <- c(0,1) # Input patterns
v1 <- c(0.6,-0.1,0.3) # Initial weight
v2 <- c(-0.3, 0.4, 0.5) # Initial weight
w <- c(0.4, 0.1, -0.2) # Initial weight
alpha = 0.25 # Learning rate
target = 1 # Target output

# Binary sigmoidal activation function
bin_sig_act_fuc <- function(x){
  act_fun = 1/(1+exp(-x))
}


### Training phase or Feed forward network

# Calculate the output for hidden unit

hidden_output <- function(input, weight1, weight2){
  sum1=0
  sum2=0
  for (i in 1:length(input)){
    prod1 = input[i]*weight1[i]
    sum1 = sum1+prod1
    prod2 = input[i]*weight2[i]
    sum2 = sum2+prod2
  }
  z = c()
  z_in1= weight1[3]+sum1
  z_in2= weight2[3]+sum2
  z[1] = bin_sig_act_fuc(z_in1)
  z[2] = bin_sig_act_fuc(z_in2)
  return(z)
}

### Training phase or Feed forward network
# We have restricted this loop to 150 iterations
for (d in 1:150) {
  z <- hidden_output(x,v1,v2)
  # Calculate the output signal from hidden layer
  output_signal <- function(input, weight1){
    sum1=0
    for (i in 1:length(z)) {
      prod1 = input[i]*weight1[i]
      sum1 = sum1+prod1
    }
    y_in = weight1[3]+sum1
    y = bin_sig_act_fuc(y_in)
  }
    y <- output_signal(z, w)
    # Stop if the difference between the expected and achieved output is less than 0.1
    if (target-y<0.1) {
      # The achieved output
      print(paste("The achieved output =", y))
      # The number of iterations required to achieve the above output
      print(paste("The number of iterations required =", d))
      break
    } else {
      ### Back propogation network
      # Error correction factor between hidden & output
      deri_y <- y*(1-y)
      delta_k <- (1-y)*deri_y
      # Find weight & bais correction term
      delta_w_jk <- c()
      for (i in 1:length(z)) {
        delta_w_jk[i] <- alpha * delta_k * z[i]
      }
      delta_w_jk[length(z)+1] <- alpha * delta_k
      delta_w_jk
      # Calculate error term between input and hidden
      delta_in_j <- c()
      deri_z_in <- c()
      delta <- c() 
      for (i in 1:length(w)-1) {
        for (j in 1:length(z)) {
          for (k in 1:length(delta_in_j)) {
            delta_in_j[i] = delta_k*w[i]
            deri_z_in[j] <- z[j]*(1-z[j])
            delta[k] = delta_in_j[k]*deri_z_in[k]
          }
        }
      }
      # Compute change in weights & bias based on delta
      delta_v1 <- c()
      delta_v2 <- c()
      for(i in 1:(length(delta)+1)) {
        delta_v1[i] = alpha * delta[1] *x[i]
        delta_v1[3] = alpha * delta[1]
      }
      for(i in 1:(length(delta)+1)) {
        delta_v2[i] = alpha * delta[2] *x[i]
        delta_v2[3] = alpha * delta[2]
      }
      delta_v1
      delta_v2
      
      # Update weight & bias on output unit
      w_new = c()
      for (i in 1:length(w)) {
        w_new[i] = w[i]+delta_w_jk[i]
      }
      w_new
      
      # Update weigth & bias on hidden unit
      v1_new = c()
      for (i in 1:length(v1)) {
        v1_new[i] = v1[i]+delta_v1[i]
      }
      v1_new
      
      v2_new = c()
      for (i in 1:length(v2)) {
        v2_new[i] = v2[i]+delta_v2[i]
      }
      v2_new
      
      # Now use the updated weight to again run the model again
      w = w_new
      v1 = v1_new
      v2 = v2_new
    }
}

function prolongSequence(sequence, size) {
  
  const Ncount = size - sequence.length;
  const leftNcount = Math.floor(Math.random() * (Ncount+1));
  const rightNcount = Ncount - leftNcount;
  const leftNs = "N".repeat(leftNcount);
  const rightNs = "N".repeat(rightNcount);
  return leftNs.concat(sequence, rightNs);
  
}


function validateSequence(sequence, size) {
  
  if (sequence.length > size) {
    return 'The sequence is too long. The sequence needs to be shorter or equal to '.concat(size, '.');
  } else if ('' != sequence.replace(/A/g,'').replace(/T/g,'').replace(/U/g,'').replace(/C/g,'').replace(/G/g,'').replace(/N/g,'')) {
    return 'The sequence must consist only of "A", "C", "T", "G", "U" and "N" characters.';
  } else {
    return '';
  }
  
}


function formatOutput(sequence, result, error) {
  
  var output = "<br><br><b>Input:</b><br><code>".concat(sequence.replace(/(.{50})/g,"$1<br>"), "</code><br><br>");
  if (error) {
    output = output.concat("<b>Error:</b><br>", result);
  } else {
    output = output.concat("<b>Output:</b><br>Probability of G4 complex =   ", result);
  }

  return output
  
}


async function makePrediction() {
  
  // get HTML elements
  var prob = document.getElementById('prob2');
  var seq = document.getElementById('text1');
  
  console.log("model loading..");

  // clear the model variable
  var model = undefined;
  
  // load model
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/simecek/penguinn_js/master/assets/model.json");
  
  console.log("model loaded...");

  // reformat the sequence and check input format
  var s = seq.value.replace(/\r?\n|\r/g,'').replace(/\s/g, '').toUpperCase();
  const validation = validateSequence(s, 200)
  if (validation != '') {
    console.log("wrong input...");
    prob.innerHTML = formatOutput(s, validation, 1);
    return 1;
  }
  if (s.length < 200) {
    var s2 = prolongSequence(s, 200);
  } else {
    var s2 = s;
  }
  
  // one-hot encoding
  const t = s2.replace(/A/g,'0').replace(/T/g,'1').replace(/U/g,'1').replace(/C/g,'2').replace(/G/g,'3').replace(/N/g,'9')
  const y = tf.oneHot(tf.tensor1d(t.split(''),'int32'),4);
  const z = y.reshape([1,200,4]);
  
  // inference
  const result = model.predict(z).asScalar().dataSync();
  console.log("prediction done...");
  
  // output
  prob.innerHTML = formatOutput(s, result, 0);
  return 0;
  
}

// Example: 
// GAGACACCACTACAGTTAGCAGTGAGTGTAAAATAATGAGTGTCAGAAACTTATATTGGGTGATTTCATTTTTAAAAGTAACCAAAGTGAAAAATGAAGCCTTGCGTTTTTGCTTAAATGATTTACAAAAAATATTTGATGTCCATCCTGGGATAGGGAATTCCTCCCCCATAACTTTGAAAGTGCAGTTGCTTCATTCC
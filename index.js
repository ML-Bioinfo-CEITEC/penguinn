function prolongSequence(sequence, size) {  
  const Ncount = size - sequence.length;
  const leftNcount = Math.floor(Math.random() * (Ncount+1));
  const rightNcount = Ncount - leftNcount;
  const leftNs = "N".repeat(leftNcount);
  const rightNs = "N".repeat(rightNcount);
  return leftNs.concat(sequence, rightNs);
}

function validateSequence(sequence, minSize=20, maxSize=200) {
  if (sequence.length > maxSize) {
    return 'The sequence is too long. The sequence needs to be shorter or equal to '.concat(maxSize, '.');
  } else if (sequence.length < minSize) {
    return 'The sequence is too short. The sequence needs to be longer or equal to '.concat(minSize, '.');
  } else if ('' != sequence.replace(/A/g,'').replace(/T/g,'').replace(/U/g,'').replace(/C/g,'').replace(/G/g,'').replace(/N/g,'')) {
    return 'The sequence must consist only of "A", "C", "T", "G", "U" and "N" characters.';
  } else {
    return '';
  }
}

function formatOutput(sequence, result, error=0, seq_name='') {
  var output = "<br/><b>Input:</b><seqtext>";
  if (seq_name!='') {
    output = output.concat(">", seq_name, "<br/>");
  }
  output = output.concat(sequence.replace(/(.{50})/g,"$1<br/>"), "</seqtext>");
  if (error) {
    output = output.concat("<b>Error:</b><br/><br/>", result, '<br/><br/><br/>');
  } else {
    if (result > 0.85) {
      output = output.concat("<b>Output:</b><br/><br/>Probability of G4 complex =   ", result, ', higher than PENGUINN Precise score threshold.<br/><br/><br/>');
    } else if (result > 0.5) {
      output = output.concat("<b>Output:</b><br/><br/>Probability of G4 complex =   ", result, ', higher than PENGUINN Sensitive score threshold.<br/><br/><br/>');
    } else {
      output = output.concat("<b>Output:</b><br/><br/>Probability of G4 complex =   ", result, ', sequence does not pass PENGUINN threshold.<br/><br/><br/>');
    }  
  }

  return output
}

function oneHot(s200) {
  // one-hot encoding
  const t = s200.replace(/A/g,'0').replace(/T/g,'1').replace(/U/g,'1').replace(/C/g,'2').replace(/G/g,'3').replace(/N/g,'9')
  const y = tf.oneHot(tf.tensor1d(t.split(''),'int32'),4);
  return y.reshape([1,200,4]);
}

function simpleSeq(x) {
  return {name: '', seq: x};
}

async function makePrediction() {
  
  // get HTML elements
  var prob = document.getElementById('prob2');
  prob.innerHTML = '';
  var txt = document.getElementById('text1').value;
  
  console.log("model loading..");

  // clear the model variable
  var model = undefined;
  // load model
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/penguinn/gh-pages/assets/model.json");
  console.log("model loaded...");

  // parse input text into the array of sequences
  var seqArray = [];
  if(document.getElementById('opt_single').checked) {
    const seq = txt.replace(/\r?\n|\r/g,'').replace(/\s/g, '');
    seqArray = [simpleSeq(seq)];
  }
  if(document.getElementById('opt_fasta').checked) {
    var fasta = require("biojs-io-fasta");
    seqArray = fasta.parse(txt);
  }  
  if(document.getElementById('opt_multiline').checked) {
    seqArray = txt.split(/\r?\n/).map(simpleSeq);
  }
  
  // process the array of sequences
  for (var i = 0; i < seqArray.length; i++) {  
    
    var s = seqArray[i].seq.toUpperCase();
    
    // check the sequence format
    const validation = validateSequence(s, 20, 200)
    if (validation != '') {
      console.log("wrong input...");
      prob.innerHTML += formatOutput(s, validation, 1, seqArray[i].name);
      continue;
    }
    var s2 = s;
    if (s.length < 200) {
      s2 = prolongSequence(s, 200);
    }  
  
    // from string to one-hot array
    const a = oneHot(s2);
  
    // inference
    const result = model.predict(a).asScalar().dataSync();
    console.log("prediction done...");
  
    // output
    prob.innerHTML += formatOutput(s, result, 0, seqArray[i].name);
  }  
  
}

// Example(s): 
// GAGACACCACTACAGTTAGCAGTGAGTGTAAAATAATGAGTGTCAGAAACTTATATTGGGTGATTTCATTTTTAAAAGTAACCAAAGTGAAAAATGAAGCCTTGCGTTTTTGCTTAAATGATTTACAAAAAATATTTGATGTCCATCCTGGGATAGGGAATTCCTCCCCCATAACTTTGAAAGTGCAGTTGCTTCATTCC
// 0.00069759286
// NNNGAAGAGACCAAGACGGAAGACCCAATCGGACCGGGAGGTCCGGGGAGACGTGTCGGGGATCGGGACTTGACTGTGCTTACCAAAGGACCTAACGGAGGGGTCCATAGGAGTCTTGCGGGACTCCCTGGCACTGGAGTAGTATCGACATAAGGGTCACGGACGTTCCATTTAGTGAGCCATTTATAAACCACTATCNN
// 0.87126887
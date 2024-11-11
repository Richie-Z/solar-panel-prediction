const pdf2doi = require("pdf2doi");

pdf2doi.fromFile("./src/test.pdf").then((doi) => {
  console.log(doi.doi);
})

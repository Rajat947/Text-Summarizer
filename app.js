const fs = require('fs');
const express = require('express');
const morgan = require('morgan');
const { urlencoded } = require('body-parser');
const { title } = require('process');
const port = 3000;
const app = express();
const { spawn } = require('child_process')
//Request Logger
app.use(morgan('tiny'));
//Body-Parser
app.use(urlencoded({extended:true}));
//Static-Files
app.use(express.static(__dirname + '/public'));
//View-Engine
app.set('view engine', 'ejs');

app.get('/' , function(req,res){
    res.render('index');
});

app.post('/', function(req,res){
    let paragraph = req.body.para;
    let title = req.body.title;
    //Python Magic -> goes here as we have the para here
    //after this magic write the summary into a file (summary.txt)
    const python = spawn('python', ['Final.py',title,paragraph])

    python.stdout.on('data', (data) => {
        console.log(data.toString())
    })

    python.on('exit', () => {
        console.log("Data Processed");
        res.redirect('summary');
    })
    //after writing redirect user to summary url
    
    
    //that's all
    
});
app.get('/summary', function(req,res){
    const data = fs.readFileSync('Text/summary.txt');

    res.render('result', {summary : data});
})
app.listen(port, (err) => {
    if(err){
        console.log(`Error Occured`);
    }else console.log(`Server is running on port ${port}`);
});
const fs = require('fs');
const express = require('express');
const morgan = require('morgan');
const { urlencoded } = require('body-parser');
const { title } = require('process');
const port = 3000;
const app = express();
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
    fs.writeFileSync('paragraph.txt', paragraph);
    //Python Magic -> goes here as we have the para here
    //after this magic write the summary into a file (summary.txt)
    let dummydata = `Lorem ipsum dolor sit amet consectetur, adipisicing elit. Ducimus, consequuntur. Tempora commodi porro impedit dicta esse, quidem accusantium cupiditate pariatur eum perferendis, officia quas aspernatur vel, praesentium doloribus natus vero.
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Eius fugit quidem accusantium sit officia quis ea blanditiis ducimus totam maxime consectetur excepturi nam expedita facilis, vel ipsa quam fuga voluptas?
    Eius fugit quidem accusantium sit officia quis ea blanditiis ducimus totam maxime consectetur excepturi nam expedita facilis, vel ipsa quam fuga voluptas?Eius fugit quidem accusantium sit officia quis ea blanditiis ducimus totam maxime consectetur excepturi nam expedita facilis, vel ipsa quam fuga voluptas?Eius fugit quidem accusantium sit officia quis ea blanditiis ducimus totam maxime consectetur excepturi nam expedita facilis, vel ipsa quam fuga voluptas?`;
    fs.writeFileSync('summary.txt', dummydata);
    
    //after writing redirect user to summary url
    
    res.redirect('summary');
    //that's all
    
});
app.get('/summary', function(req,res){
    const data = fs.readFileSync('summary.txt');

    res.render('result', {summary : data});
})
app.listen(port, (err) => {
    if(err){
        console.log(`Error Occured`);
    }else console.log(`Server is running on port ${port}`);
});
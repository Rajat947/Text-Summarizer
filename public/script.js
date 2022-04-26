const submitBtn = document.querySelector('.btn');
const paraInp = document.querySelector('.para-inp');
const titleInp = document.querySelector('.para-title-inp');
const bodyEl = document.body;
const backdrop = document.querySelector('.backdrop');
const renderError = function(err){
    const html = `
    <div class="errormsg">${err}</div>
    `;
    bodyEl.insertAdjacentHTML('afterbegin', html);
    setTimeout(() => {
        const errorEL = document.querySelector('.errormsg');
        errorEL.remove();
    }, 3000);
}
submitBtn.addEventListener('click', function(e){
    const title = titleInp.value;
    const para = paraInp.value;
    const num_words = para.split(' ').length;
    console.log(num_words);
    if(title === "" && para !== ""){
        renderError("Title is Required!");
        e.preventDefault();
        return;
    }
    if(para === "" && title !== ""){
        renderError("Paragraph is Required!");
        e.preventDefault();
        return;
    }
    if(title === "" && para === ""){
        renderError("Cannot Submit Empty Form!");
        e.preventDefault();
        return;
    }
    if(num_words < 100){
        renderError("Paragraph should have min-words : 100");
        e.preventDefault();
        return;
    }
    //show backdrop and  show loader ->
    if(backdrop.classList.contains('hide')){
        backdrop.classList.remove('hide');
    }
});
bodyEl.addEventListener('load', function(){
    backdrop.classList.add('hide');
})
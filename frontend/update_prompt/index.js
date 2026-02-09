import jsyaml from 'https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/+esm';

async function loadPrompt() {
    // ใช้ absolute path จาก root 
    const timestamp = new Date().getTime();
    const response = await fetch(`/code/prompt/prompt_mockdata.yaml?t=${timestamp}`);
    const yamlText = await response.text();
    const data = jsyaml.load(yamlText);
    
    $("#prompt-code-agent").text(data.prompt_mockdata); 
}

loadPrompt();

async function savePrompt(){
    const newPrompt = $("#prompt-code-agent").val();
    const response = await fetch("api/update-prompt",{
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({prompt: newPrompt})
    });
    
    if (response.ok){
        $("#success-message").text("บันทึกข้อมูลสำเร็จแล้ว")
                            .removeClass("error-box")
                            .addClass("success-box")
                            .fadeIn();

        loadPrompt(); 

        setTimeout(() => {
            $("#success-message").fadeOut();
        }, 1500);
    }
    else{
        $("#success-message").text("✗ บันทึกไม่สำเร็จ")
                  .removeClass("success-box")
                  .addClass("error-box")
                  .fadeIn();

        loadPrompt();

        setTimeout(() => {
            $("#success-message").fadeOut();
        }, 1500);
    }

}

$("#uppdate-prompt-code-agent").click(savePrompt);


function login() {
    var userType = document.getElementById("userType").value;
    var userType = document.getElementById("userType").value;
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;
    
    if (userType === "admin" && username === "admin" && password === "password123") {
        alert("Admin login successful!");
        window.location.href = "admin_dashboard.html";
    } else if (userType === "staff" && username === "staff" && password === "staff123") {
        alert("Staff login successful!");
        window.location.href = "staff_dashboard.html";
    }else if (userType === "student" && username === "student" && password === "stud123") {
        alert("Student login successful!");
        window.location.href = "student_dashboard.html";
    }
     else {
        document.getElementById("error-message").innerText = "Invalid credentials!";
    }
}
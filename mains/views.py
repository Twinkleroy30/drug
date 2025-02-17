from django.shortcuts import render, redirect

def index(request):
    return render(request, 'main/index.html')

def about(req):
    return render(req, 'main/about.html')

def buy(req):
    return render(req, 'main/buy.html')

def contact(req):
    return render(req, 'main/contact.html')

def medicine(req):
    return render(req, 'main/medicine.html')


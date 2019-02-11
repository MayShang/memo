// table of contents
var ToC =
  "<nav role='navigation' class='table-of-contents'>" +
    "<h6>Contents :</h6>" +
    "<ul>";

var newLine, el, title, link;

$("h3").each(function() {

  el = $(this);
  title = el.text();
  link = "#" + el.attr("id");

  newLine =
    "<li>" +
      "<a href='" + link + "'>" +
        title +
      "</a>" +
    "</li>";

  // newLine = $("<li>").append( $("<a>").attr('href', link).text(title) );

  ToC += newLine;

});

ToC +=
   "</ul>" +
  "</nav>";

$(".toc-area").prepend(ToC);

// list item slideToggle to hide and shows up
$(".list-item").click(function(){
  $(this).children().not("span.badge").slideToggle("fast", function(){});
});

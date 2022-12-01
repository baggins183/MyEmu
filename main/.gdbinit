set history save on
display/i $pc

define bsave
   set logging file .brestore.txt
   set logging on
   info break
   set logging off
   # Reformat on-the-fly to a valid gdb command file
   shell perl -n -e 'print "break $1\n" if /^\d+.+?(\S+)$/g' .brestore.txt > .brestore.gdb
   shell rm -f .brestore.txt
end
document bsave
  store actual breakpoints
end

define brestore
  set breakpoint pending on
  source .brestore.gdb
end
document brestore
  restore breakpoints saved by bsave
end

brestore
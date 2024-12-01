local foo = pd.Class:new():register("foo")


function foo:initialize(sel, atoms)

   self.inlets = 1

   self.outlets = 1

   if type(atoms[1]) == "number" then

      self.counter = atoms[1]

   else

      self.counter = 0

   end

   if type(atoms[2]) == "number" then

      self.step = atoms[2]

   else

      self.step = 1

   end

   return true

end


function foo:in_1_bang()

   self:outlet(1, "float", {self.counter})

   self.counter = self.counter + self.step

end